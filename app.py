from flask import Flask, render_template, request, send_file, Response
import os
import io
import large_image
import object_detection.main as object_detection
import classification.main as classification

app = Flask(__name__)
slides_root_dir = "data/whole-slides/gut"


def get_slide_path(slide_name):
    return os.path.join(slides_root_dir, f"{slide_name}.svs")


@app.route('/')
def index():
    """Home page listing all available slides."""
    slides = [f[:-4] for f in os.listdir(slides_root_dir) if f.endswith(".svs")]
    return render_template('index.html', slides=slides)


@app.route('/slide/<slide_name>')
def view_slide(slide_name):
    """Page to view a specific slide."""
    slide_path = get_slide_path(slide_name)
    if not os.path.exists(slide_path):
        return "Slide not found", 404

    tile_source = large_image.open(slide_path)
    metadata = tile_source.getMetadata()
    print(metadata)
    return render_template('viewer.html', slide_name=slide_name,width=metadata["sizeX"],height=metadata["sizeY"],tileSize=metadata["tileWidth"], **metadata)


@app.route('/tile/<slide_name>')
def get_tile(slide_name):
    """Serves dynamic tiles for a given slide."""
    slide_path = get_slide_path(slide_name)
    if not os.path.exists(slide_path):
        return "Slide not found", 404

    x, y, z = map(int, [request.args.get(k) for k in ('x', 'y', 'z')])
    tile_source = large_image.open(slide_path)
    tile_image = tile_source.getTile(x, y, z, format='PNG')
    return send_file(io.BytesIO(tile_image), mimetype="image/png")


@app.route('/stream/<method>/<slide_name>')
def stream_results(method, slide_name):
    """Streams results from object detection or classification."""
    slide_path = get_slide_path(slide_name)
    if not os.path.exists(slide_path):
        return "Slide not found", 404

    if method == "object-detection":
        return Response(object_detection.process_slide(slide_path), mimetype='text/event-stream')
    elif method == "classification":
        return Response(classification.process_slide(slide_path), mimetype='text/event-stream')
    else:
        return "Invalid method", 400


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
