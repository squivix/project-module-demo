import base64
import io
import json

import numpy as np
import torch
from PIL import Image, ImageDraw
from monai.data import Dataset, DataLoader
from monai.inferers import AvgMerger, PatchInferer, WSISlidingWindowSplitter
from ultralytics import YOLO

model = None


def load_yolo_model():
    global model
    if model is None:
        model_path = "object_detection/saved-models/best.pt"
        model = YOLO(model_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
    return model


def filter_fn(patch, loc):
    """Filter patches based on white space content"""
    patch_np = patch[0].cpu().numpy()
    white_pixels = np.sum(patch_np > 240) / patch_np.size
    return white_pixels <= 0.75


def process_slide(slide_path):
    model = load_yolo_model()
    wsi_level = 0
    patch_size = 640

    inferer = PatchInferer(
        splitter=WSISlidingWindowSplitter(
            patch_size=patch_size,
            overlap=0.0,
            pad_mode=None,
            reader="openslide",
            level=wsi_level,
            filter_fn=filter_fn
        ),
        merger_cls=AvgMerger,
        match_spatial_shape=True,
    )

    for inputs in DataLoader(Dataset([slide_path])):
        for patch, location in inferer.splitter(inputs):
            if filter_fn(patch, location):
                img_np = patch[0].permute(1, 2, 0).cpu().numpy()
                img_pil = Image.fromarray(img_np.astype(np.uint8))
                results = model.predict(
                    source=img_pil,
                    conf=0.15,
                    agnostic_nms=True,
                    max_det=5,
                    show_labels=False,
                    augment=False,
                    verbose=False,
                    show_conf=False
                )

                valid_boxes = []
                boxes_data = []
                img_width, img_height = img_pil.size
                confidence = 0.0
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height

                    if width <= 0.5 and height <= 0.5:
                        valid_boxes.append(box)
                        boxes_data.append({
                            'x1': x1 / img_width,
                            'y1': y1 / img_height,
                            'x2': x2 / img_width,
                            'y2': y2 / img_height,
                            'conf': float(box.conf[0]),
                            'cls': int(box.cls[0])
                        })
                        confidence = max(confidence, box.conf[0].item())

                if valid_boxes:
                    draw_img = img_pil.copy()
                    draw = ImageDraw.Draw(draw_img)

                    for box in valid_boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)

                    buffered = io.BytesIO()
                    draw_img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    response_data = {
                        "region": {
                            'image': img_str,
                            'location': list(location[::-1]) + [patch_size, patch_size],
                            'boxes': boxes_data,
                            "confidence": confidence,
                        }
                    }
                    yield f"data: {json.dumps(response_data)}\n\n"
