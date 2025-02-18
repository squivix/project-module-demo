import base64
import io
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from classification.datasets.SlideSeperatedImageDataset import SlideSeperatedImageDataset
from classification.extractors.TemplateMatchExtractor import TemplateMatchExtractor, generate_dataset_from_slides
from classification.labelers.GroundTruthLabeler import GroundTruthLabeler
from classification.models.mlp import MLPBinaryClassifier
from classification.models.resnet import Resnet101BinaryClassifier


def tensor_to_base64_png(tensor):
    # Ensure the tensor has 3 channels (RGB) and is in the range [0, 1]
    if tensor.ndimension() == 3 and tensor.size(0) == 3:
        tensor = tensor.permute(1, 2, 0)  # Convert from CxHxW to HxWxC

    # Check if tensor is in the range [0, 1] (float)
    if tensor.max() <= 1.0:
        tensor = tensor.mul(255).byte()  # Scale to 0-255 and convert to byte tensor
    else:
        tensor = tensor.clamp(0, 255).byte()  # Ensure the values are within [0, 255]

    # Convert tensor to a PIL Image
    pil_image = Image.fromarray(tensor.numpy())  # Convert tensor to PIL image

    # Save the image to a BytesIO buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")

    # Encode the image in base64
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return img_str


def process_slide(slide_path):
    slide_name = Path(slide_path).stem
    PretrainedModelClass = Resnet101BinaryClassifier
    pretrained_model_name = PretrainedModelClass.get_pretrained_model_name()
    pretrained_output_size = PretrainedModelClass.pretrained_output_size
    slides_root_dir = "data/whole-slides/gut"
    labels_root_dir = "data/labels"
    candidates_dataset_dir = "classification/temp/candidates"
    model_output_dir = "classification/saved-models"

    ground_truth_labeler = GroundTruthLabeler(f"{labels_root_dir}/slide-annotations/all.json",
                                              f"{labels_root_dir}/patch-classifications.csv")
    extractor = TemplateMatchExtractor(ground_truth_labeler)
    generate_dataset_from_slides(slides_root_dir, extractor, candidates_dataset_dir, slide_filenames=[f"{slide_name}.svs"])
    dataset = SlideSeperatedImageDataset(candidates_dataset_dir, {slide_name}, with_index=True)
    batch_size = 256
    threshold = 0.3
    state_dict = torch.load(f"{model_output_dir}/{pretrained_model_name}.pickle", weights_only=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    classifier = MLPBinaryClassifier(in_features=pretrained_output_size, hidden_layers=1,
                                     units_per_layer=2048,
                                     dropout=0.1, focal_alpha=0.75, focal_gamma=2.5)
    classifier.load_state_dict(state_dict)
    model = PretrainedModelClass(model=classifier).to(device)
    test_loader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    all_y_probs = []

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(tqdm(test_loader, desc=f"Classifying")):
            batch_x = batch[0].to(device)
            batch_indexes = batch[2].to(device)
            batch_y_probs = model.forward(batch_x)
            all_y_probs.append(batch_y_probs.cpu().detach().numpy())
            for j in range(len(batch_indexes)):
                file_path = dataset.get_item_file_path(batch_indexes[j])
                confidence = batch_y_probs[j].item()
                if confidence < threshold:
                    continue
                patch_bbox = Path(file_path).stem.split("_")[1:]
                original_image, _ = dataset.get_item_untransformed(batch_indexes[j])
                response_data = {
                    "region": {
                        'image': tensor_to_base64_png(original_image),
                        'location': patch_bbox,
                        "confidence": confidence,
                    },
                    "progress": {
                        "currentStep": 2,
                        "totalStep": 2,
                        "stepName": "Classifying",
                        "stepProgress": i / len(test_loader),
                    }
                }
                yield f"data: {json.dumps(response_data)}\n\n"
    response_data = {
        "progress": {
            "currentStep": 2,
            "totalStep": 2,
            "stepName": "Classifying",
            "stepProgress": 1,
        }
    }
    yield f"data: {json.dumps(response_data)}\n\n"
