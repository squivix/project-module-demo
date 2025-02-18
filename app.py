import io
import os
from threading import Lock

import large_image
from flask import Flask, render_template, request, send_file, Response, jsonify

import classification.main as classification
import object_detection.main as object_detection

app = Flask(__name__)
slides_root_dir = "data/whole-slides/gut"
active_jobs = {}
lock = Lock()


def get_slide_path(slide_name):
    return os.path.join(slides_root_dir, f"{slide_name}.svs")


@app.route('/')
def index():
    """Home page listing all available slides."""
    slides = sorted([f[:-4] for f in os.listdir(slides_root_dir) if f.endswith(".svs")])
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
    return render_template('viewer.html', slide_name=slide_name, width=metadata["sizeX"], height=metadata["sizeY"], tileSize=metadata["tileWidth"], **metadata)


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


method_map = {
    "object-detection": object_detection.process_slide,
    "classification": classification.process_slide,
}


@app.route('/stream/<method>/<slide_name>')
def stream_results(method, slide_name):
    slide_path = get_slide_path(slide_name)
    if not os.path.exists(slide_path):
        return "Slide not found", 404
    if method not in ["object-detection", "classification"]:
        return "Invalid method", 400

    client_id = request.remote_addr  # Using client IP as identifier; consider more reliable methods
    job_key = (client_id, method, slide_name)

    with lock:
        if job_key in active_jobs:
            return jsonify({"error": "Job already in progress for this client"}), 429  # 429 Too Many Requests
        else:
            active_jobs[job_key] = True

    def generate():
        try:
            for data in method_map[method](slide_path):
                yield data
        finally:
            with lock:
                active_jobs.pop(job_key, None)

    return Response(generate(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
