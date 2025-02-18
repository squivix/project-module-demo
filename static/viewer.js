let viewer;
let objectDetectionResults = [];
let classificationResults = [];
let classificationProgress = null;
let objectDetectionProgress = null;

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
}

function renderResults() {
    const container = document.getElementById("results-container");
    container.querySelectorAll(".result-entry").forEach(n => n.remove())
    const results = getTabData().results;
    const emptyMessage = document.getElementById("empty-message");

    if (results.length === 0) {
        emptyMessage.style.display = "flex";
        if (currentTab === TAB_ENUM.CLASSIFICATION) {
            emptyMessage.querySelector("#object-detection").style.display = "none"
            emptyMessage.querySelector("#classification").style.display = "block"
        } else {
            emptyMessage.querySelector("#classification").style.display = "none"
            emptyMessage.querySelector("#object-detection").style.display = "block"
        }
    } else {
        emptyMessage.style.display = "none";
        results.forEach(data => {
            const entry = document.createElement("div");
            entry.className = "result-entry";
            entry.innerHTML = `<img src='data:image/png;base64,${data.image}' class='result-image'>
                               <p>Confidence: ${data.confidence.toFixed(2)}</p>`;
            container.appendChild(entry);
        });
    }
}

function startStreaming(method, slideName) {
    changeTab(method);
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

function handleStreamData(event, method) {
    try {
        const streamData = JSON.parse(event.data);
        if (streamData.region) {
            const results = method === TAB_ENUM.OBJECT_DETECTION ? objectDetectionResults : classificationResults;
            results.push({...streamData.region});
        }
        if (streamData.progress)
            updateProgress(method, streamData.progress)
        renderResults();
    } catch (error) {
        console.error("Error processing streamed data:", error);
    }
}
