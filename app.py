import io
import os
from pathlib import Path

import large_image
from flask import Flask, send_file, request, render_template, Response

import object_detection.main as object_detection
import classification.main as classification

app = Flask(__name__)

# Root directory where slides are stored
slides_root_dir = "data/whole-slides/gut"


@app.route('/')
def index():
    """Home page listing all available slides."""
    slides = [
        Path(f).stem for f in os.listdir(slides_root_dir) if f.endswith(".svs")
    ]
    return render_template('index.html', slides=slides)


@app.route('/slide/<slide_name>')
def view_slide(slide_name):
    """Page to view a specific slide."""
    slide_path = os.path.join(slides_root_dir, f"{slide_name}.svs")

    if not os.path.exists(slide_path):
        return "Slide not found", 404

    tile_source = large_image.open(slide_path)
    metadata = tile_source.getMetadata()

    return render_template('viewer.html',
                           slide_name=slide_name,
                           height=metadata['sizeY'],
                           width=metadata['sizeX'],
                           tileSize=metadata['tileWidth'],
                           levels=metadata['levels']
                           )


@app.route('/tile/<slide_name>')
def get_tile(slide_name):
    """Dynamically serves tiles for a given slide."""
    slide_path = os.path.join(slides_root_dir, f"{slide_name}.svs")

    if not os.path.exists(slide_path):
        return "Slide not found", 404

    x = int(request.args.get('x'))
    y = int(request.args.get('y'))
    z = int(request.args.get('z'))
    tile_source = large_image.open(slide_path)
    tile_image = tile_source.getTile(x, y, z, format='PNG')
    return send_file(io.BytesIO(tile_image), mimetype="image/png")


@app.route('/object-detection/<slide_name>')
def object_detection_stream(slide_name):
    """Stream object detection results."""
    slide_path = os.path.join(slides_root_dir, f"{slide_name}.svs")
    print("slide_path", slide_path)
    if not os.path.exists(slide_path):
        return "Slide not found", 404

    return Response(
        object_detection.process_slide(slide_path),
        mimetype='text/event-stream'
    )


@app.route('/classification/<slide_name>')
def classification_stream(slide_name):
    """Stream classification results."""
    slide_path = os.path.join(slides_root_dir, f"{slide_name}.svs")
    print("slide_path", slide_path)
    if not os.path.exists(slide_path):
        return "Slide not found", 404

    return Response(
        classification.process_slide(slide_path),
        mimetype='text/event-stream'
    )


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
