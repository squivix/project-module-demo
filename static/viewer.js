let viewer;
let objectDetectionResults = [];
let classificationResults = [];

const RESULTS_TYPE = {OBJECT_DETECTION: "object-detection", CLASSIFICATION: "classification"}
let activeResults = RESULTS_TYPE.OBJECT_DETECTION;
let RESULTS_TO_SHOW = 5; // Number of results to render initially

function initializeViewer(slideName, metadata) {
    return OpenSeadragon({
        id: "openseadragon",
        prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/2.4.2/images/",
        tileSources: {
            width: metadata.width,
            height: metadata.height,
            tileWidth: metadata.tileSize,
            tileHeight: metadata.tileSize,
            minLevel: 0,
            maxLevel: metadata.levels - 1,
            getTileUrl: (level, x, y) => `/tile/${slideName}?z=${level}&x=${x}&y=${y}`
        },
        showNavigator: true,
        navigatorPosition: "BOTTOM_RIGHT"
    });
}

function getViewportRect(x, y, w, h, full_w, full_h) {
    const aspectRatio = full_w / full_h;
    const bbox = {
        x: x / full_w,
        y: (y / (full_h * aspectRatio)),
        w: w / full_w,
        h: (h / (full_h * aspectRatio))
    };
    return new OpenSeadragon.Rect(bbox.x, bbox.y, bbox.w, bbox.h);
}

function addPatchOverlay(location, boxes) {
    const [absx, absy, width, height] = location;
    const patchOverlay = getViewportRect(absx, absy, width, height, window.slideMetadata.width, window.slideMetadata.height);
    const container = document.createElement('div');
    container.className = 'patch-overlay-container';

    const patchOutline = document.createElement('div');
    patchOutline.className = 'highlight-overlay';
    if (boxes)
        patchOutline.style.border = '2px solid green';
    else
        patchOutline.style.border = '4px solid red';
    patchOutline.style.width = '100%';
    patchOutline.style.height = '100%';
    container.appendChild(patchOutline);

    boxes?.forEach(box => {
        const boxElement = document.createElement('div');
        boxElement.className = 'yolo-box-overlay';
        boxElement.style.position = 'absolute';
        boxElement.style.border = '2px solid red';
        boxElement.style.background = 'rgba(255, 0, 0, 0.1)';

        boxElement.style.left = `${box.x1 * 100}%`;
        boxElement.style.top = `${box.y1 * 100}%`;
        boxElement.style.width = `${(box.x2 - box.x1) * 100}%`;
        boxElement.style.height = `${(box.y2 - box.y1) * 100}%`;

        container.appendChild(boxElement);
    });

    viewer.addOverlay(container, patchOverlay, OpenSeadragon.OverlayPlacement.TOP_LEFT);
    return patchOverlay;
}

function startStreaming(method, slideName) {
    changeResults(method);
    const eventSource = new EventSource(`/stream/${method}/${slideName}`);
    eventSource.onmessage = event => handleStreamData(event, method);
    eventSource.onerror = () => eventSource.close();
}

function handleStreamData(event, method) {
    try {
        const data = JSON.parse(event.data);
        let results;
        if (method === RESULTS_TYPE.OBJECT_DETECTION)
            results = objectDetectionResults
        else
            results = classificationResults;
        results.push({
            ...data,
            renderedOverlayRect: null,
        });
        results.sort((a, b) => b.confidence - a.confidence);
        renderResults();
    } catch (error) {
        console.error('Error processing streamed data:', error);
    }
}

function renderResults() {
    const container = document.getElementById('results-container');
    container.innerHTML = '';
    const results = activeResults === RESULTS_TYPE.OBJECT_DETECTION ? objectDetectionResults : classificationResults;
    results.forEach((data, index) => {
        if (!data.renderedOverlayRect)
            data.renderedOverlayRect = addPatchOverlay(data.location, data.boxes)
        if (index <= RESULTS_TO_SHOW) {
            const entry = document.createElement('div');
            entry.className = 'result-entry';
            entry.innerHTML = `<img src='data:image/png;base64,${data.image}' class='result-image'>
                           <p>Confidence: ${data.confidence.toFixed(2)}</p>`;
            entry.onclick = () => viewer.viewport.fitBounds(data.renderedOverlayRect);
            container.appendChild(entry);
        }
    })
}

function changeResults(newResults) {
    activeResults = newResults;
    clearOverlays();
    renderResults();
}

function clearOverlays() {
    const results = activeResults === RESULTS_TYPE.OBJECT_DETECTION ? objectDetectionResults : classificationResults;
    viewer.clearOverlays();
    results.forEach(data => data.renderedOverlayRect = null);
}

document.addEventListener('DOMContentLoaded', () => {
    const metadata = window.slideMetadata;
    viewer = initializeViewer(metadata.slideName, metadata);
    const container = document.getElementById('results-container');

    document.getElementById('object-detection').onclick = () => startStreaming(RESULTS_TYPE.OBJECT_DETECTION, metadata.slideName);
    document.getElementById('classification').onclick = () => startStreaming(RESULTS_TYPE.CLASSIFICATION, metadata.slideName);
    document.getElementById('toggle-results').onclick = () => changeResults(activeResults === RESULTS_TYPE.OBJECT_DETECTION ? RESULTS_TYPE.CLASSIFICATION : RESULTS_TYPE.OBJECT_DETECTION);

    container.addEventListener('scroll', () => {
        if (container.scrollTop + container.clientHeight >= container.scrollHeight - 50) {
            RESULTS_TO_SHOW += 5;
            renderResults();
        }
    });
});
