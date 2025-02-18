let viewer;
let objectDetectionResults = [];
let classificationResults = [];
let classificationProgress = null;
let objectDetectionProgress = null;

const TAB_ENUM = {OBJECT_DETECTION: "object-detection", CLASSIFICATION: "classification"}
const TAB_TO_TITLE = {[TAB_ENUM.OBJECT_DETECTION]: "Object Detection", [TAB_ENUM.CLASSIFICATION]: "Classification"}
let currentTab = TAB_ENUM.OBJECT_DETECTION;
let RESULTS_TO_SHOW = 5; // Number of results to render initially

function getTabData() {
    if (currentTab === TAB_ENUM.OBJECT_DETECTION) {
        return {
            results: objectDetectionResults,
            progress: objectDetectionProgress
        }
    } else {
        return {
            results: classificationResults,
            progress: classificationProgress
        }
    }
}

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
    changeTab(method);
    const eventSource = new EventSource(`/stream/${method}/${slideName}`);
    eventSource.onmessage = event => handleStreamData(event, method);
    eventSource.onerror = () => eventSource.close();
}

function handleStreamData(event, method) {
    try {
        const streamData = JSON.parse(event.data);
        if (streamData.region) {
            let results;
            if (method === TAB_ENUM.OBJECT_DETECTION)
                results = objectDetectionResults
            else
                results = classificationResults;
            results.push({
                ...streamData.region,
                renderedOverlayRect: null,
            });
            results.sort((a, b) => b.confidence - a.confidence);
        }
        if (streamData.progress)
            updateProgress(method, streamData.progress)
        renderResults();
    } catch (error) {
        console.error('Error processing streamed data:', error);
    }
}

function renderResults() {
    const container = document.getElementById('results-container');
    container.innerHTML = '';
    const results = getTabData().results;
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

function changeTab(newTab) {
    currentTab = newTab;
    document.querySelector("#results-title").textContent = `${TAB_TO_TITLE[currentTab]} Results`
    clearOverlays();
    renderResults();
    let showProgress = getTabData().progress == null;
    const progressContainer = document.getElementById("progress-container");
    progressContainer.style.display = showProgress ? "none" : "flex";
}

function clearOverlays() {
    const results = getTabData().results
    viewer.clearOverlays();
    results.forEach(data => data.renderedOverlayRect = null);
}

function updateProgress(method, progressData) {
    if (method === TAB_ENUM.OBJECT_DETECTION)
        objectDetectionProgress = progressData;
    else
        classificationProgress = progressData;

    const progressBar = document.getElementById("progress-bar");
    const progressText = document.getElementById("progress-text");
    const currentStep = progressData.currentStep;
    const totalSteps = progressData.totalStep;
    const stepName = progressData.stepName;
    const stepProgress = progressData.stepProgress;

    const progressPercent = Math.round(stepProgress * 100);
    // Update the UI
    progressBar.style.width = progressPercent + "%";
    progressBar.textContent = progressPercent + "%";
    progressText.textContent = `(${currentStep}/${totalSteps}) ${stepName}: (${progressPercent}%)`;

    let showProgress = getTabData().progress == null;
    const progressContainer = document.getElementById("progress-container");
    progressContainer.style.display = showProgress ? "none" : "flex";
}

document.addEventListener('DOMContentLoaded', () => {
    const metadata = window.slideMetadata;
    viewer = initializeViewer(metadata.slideName, metadata);
    const container = document.getElementById('results-container');

    document.getElementById('object-detection').onclick = () => startStreaming(TAB_ENUM.OBJECT_DETECTION, metadata.slideName);
    document.getElementById('classification').onclick = () => startStreaming(TAB_ENUM.CLASSIFICATION, metadata.slideName);
    document.getElementById('toggle-results').onclick = () => changeTab(currentTab === TAB_ENUM.OBJECT_DETECTION ? TAB_ENUM.CLASSIFICATION : TAB_ENUM.OBJECT_DETECTION);

    container.addEventListener('scroll', () => {
        if (container.scrollTop + container.clientHeight >= container.scrollHeight - 50) {
            RESULTS_TO_SHOW += 5;
            renderResults();
        }
    });
});
