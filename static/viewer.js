let viewer;
let objectDetectionResults = [];
let classificationResults = [];
let classificationProgress = null;
let objectDetectionProgress = null;
let RESULTS_TO_SHOW = 5; // Number of results to render initially

const TAB_ENUM = {OBJECT_DETECTION: "object-detection", CLASSIFICATION: "classification"};
const TAB_TO_TITLE = {
    [TAB_ENUM.OBJECT_DETECTION]: "Object Detection",
    [TAB_ENUM.CLASSIFICATION]: "Classification"
};
let currentTab = TAB_ENUM.CLASSIFICATION;

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

function getTabData() {
    return currentTab === TAB_ENUM.OBJECT_DETECTION
        ? {results: objectDetectionResults, progress: objectDetectionProgress}
        : {results: classificationResults, progress: classificationProgress};
}

document.addEventListener("DOMContentLoaded", () => {
    const metadata = window.slideMetadata;
    viewer = initializeViewer(metadata.slideName, metadata);

    document.getElementById("tab-object-detection").onclick = () => changeTab(TAB_ENUM.OBJECT_DETECTION);
    document.getElementById("tab-classification").onclick = () => changeTab(TAB_ENUM.CLASSIFICATION);
    document.getElementById("object-detection").onclick = () => startStreaming(TAB_ENUM.OBJECT_DETECTION, metadata.slideName);
    document.getElementById("classification").onclick = () => startStreaming(TAB_ENUM.CLASSIFICATION, metadata.slideName);
    const container = document.getElementById("results-container")
    container.addEventListener('scroll', () => {
        if (container.scrollTop + container.clientHeight >= container.scrollHeight - 50) {
            RESULTS_TO_SHOW += 5;
            renderResults();
        } else if (container.scrollTop === 0) {
            RESULTS_TO_SHOW = 5;
            renderResults();
        }
    });
    changeTab(currentTab);
    renderResults();
});

function changeTab(newTab) {
    currentTab = newTab;
    document.querySelectorAll(".tab-button").forEach(button => button.classList.remove("active"));
    document.getElementById(`tab-${newTab}`).classList.add("active");
    document.getElementById("results-title").textContent = `${TAB_TO_TITLE[currentTab]} Results`;
    clearOverlays();
    renderResults();

    const progressContainer = document.getElementById("progress-container");
    progressContainer.style.display = !!getTabData().progress ? "flex" : "none";
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

function renderResults() {
    const container = document.getElementById("results-container");
    container.querySelectorAll(".result-entry").forEach(n => n.remove())
    const results = getTabData().results;

    const emptyMessage = document.getElementById("empty-message");
    const loadingMessage = document.getElementById("loading-message");
    if (results.length === 0) {
        if (getTabData().progress != null) {
            emptyMessage.style.display = "none";
            loadingMessage.style.display = "grid";
        } else {
            emptyMessage.style.display = "flex";
            loadingMessage.style.display = "none";
            if (currentTab === TAB_ENUM.CLASSIFICATION) {
                emptyMessage.querySelector("#object-detection").style.display = "none"
                emptyMessage.querySelector("#classification").style.display = "block"
            } else {
                emptyMessage.querySelector("#classification").style.display = "none"
                emptyMessage.querySelector("#object-detection").style.display = "block"
            }
        }
    } else {
        loadingMessage.style.display = "none";
        emptyMessage.style.display = "none";
        results.forEach((data, index) => {
            if (!data.renderedOverlayRect)
                data.renderedOverlayRect = addPatchOverlay(data.location, data.boxes)
            if (index > RESULTS_TO_SHOW)
                return;
            const entry = document.createElement("div");
            entry.onclick = () => viewer.viewport.fitBounds(data.renderedOverlayRect);
            entry.className = "result-entry";
            entry.innerHTML = `<img src='data:image/png;base64,${data.image}' class='result-image'>
                               <p>Confidence: ${data.confidence.toFixed(2)}</p>`;
            container.appendChild(entry);
        })
    }
}

function startStreaming(method, slideName) {
    changeTab(method);
    updateProgress(method, {stepProgress: 0.0})
    renderResults()
    const eventSource = new EventSource(`/stream/${method}/${slideName}`);
    eventSource.onmessage = event => handleStreamData(event, method);
    eventSource.onerror = () => eventSource.close();
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
    if (progressData) {
        const progressBar = document.getElementById("progress-bar");
        const progressText = document.getElementById("progress-text");
        const currentStep = progressData.currentStep ?? 0;
        const totalSteps = progressData.totalStep ?? "?";
        const stepName = progressData.stepName ?? "Pending...";
        const stepProgress = progressData.stepProgress;

        const progressPercent = Math.round(stepProgress * 100);
        // Update the UI
        progressBar.style.width = progressPercent + "%";
        progressBar.textContent = progressPercent + "%";
        progressText.textContent = `(${currentStep}/${totalSteps}) ${stepName}: (${progressPercent}%)`;
    }
    let showProgress = getTabData().progress == null;
    const progressContainer = document.getElementById("progress-container");
    progressContainer.style.display = showProgress ? "none" : "flex";
}

function handleStreamData(event, method) {
    try {
        const streamData = JSON.parse(event.data);
        if (streamData.region) {
            const results = method === TAB_ENUM.OBJECT_DETECTION ? objectDetectionResults : classificationResults;
            results.push({...streamData.region});
            results.sort((a, b) => b.confidence - a.confidence);
        }
        // if (streamData.progress)
        updateProgress(method, streamData.progress)
        renderResults();
    } catch (error) {
        console.error("Error processing streamed data:", error);
    }
}
