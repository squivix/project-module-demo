// viewer.js
let viewer;

function initializeViewer(slideName, height, width, tileSize, levels) {
    return OpenSeadragon({
        id: "openseadragon",
        prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/2.4.2/images/",
        tileSources: {
            width: width,
            height: height,
            tileWidth: tileSize,
            tileHeight: tileSize,
            minLevel: 0,
            maxLevel: levels - 1,
            getTileUrl: function(level, x, y) {
                return `/tile/${slideName}?z=${level}&x=${x}&y=${y}`;
            },
            overlays: [] 
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
    const patchSize = 640;
    const absx = location[0];
    const absy = location[1];

    // Create patch overlay
    const patchOverlay = getViewportRect(
        absx, 
        absy, 
        patchSize, 
        patchSize, 
        window.slideMetadata.width, 
        window.slideMetadata.height
    );

    const container = document.createElement('div');
    container.className = 'patch-overlay-container';
    container.style.position = 'absolute';
    container.style.width = '100%';
    container.style.height = '100%';

    const patchOutline = document.createElement('div');
    patchOutline.className = 'highlight-overlay';
    patchOutline.style.border = '2px solid green';
    patchOutline.style.background = 'rgba(0, 255, 0, 0)';
    patchOutline.style.position = 'absolute';
    patchOutline.style.width = '100%';
    patchOutline.style.height = '100%';
    container.appendChild(patchOutline);

    if (boxes && boxes.length > 0) {
        boxes.forEach(box => {
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
    }

    viewer.addOverlay(container, patchOverlay, OpenSeadragon.OverlayPlacement.TOP_LEFT);
    return patchOverlay;
}

function handleObjectDetection(slideName) {
    let resultsContainer = document.getElementById('detection-results');
    if (!resultsContainer) {
        resultsContainer = document.createElement('div');
        resultsContainer.id = 'detection-results';
        resultsContainer.style.cssText = 'position: fixed; right: 0; top: 0; width: 300px; height: 100vh; overflow-y: auto; background: white; padding: 10px; border-left: 1px solid #ccc;';
        document.body.appendChild(resultsContainer);
    }

    // Clear previous results and overlays
    resultsContainer.innerHTML = '<h3>Detection Results</h3>';

    const eventSource = new EventSource(`/object-detection/${slideName}`);

    eventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);

            // Create new image element
            const imgContainer = document.createElement('div');
            imgContainer.style.marginBottom = '10px';
            imgContainer.style.cursor = 'pointer';

            const img = document.createElement('img');
            img.src = `data:image/png;base64,${data.image}`;
            img.style.width = '100%';

            const location = document.createElement('p');
            location.textContent = `Location: (${data.location[0]}, ${data.location[1]})`;
            const overlayRect = addPatchOverlay(data.location, data.boxes);

            imgContainer.onclick = () => {
                const bounds = new OpenSeadragon.Rect(
                    overlayRect.x,
                    overlayRect.y,
                    overlayRect.width,
                    overlayRect.height
                );
                viewer.viewport.fitBounds(bounds);
            };

            imgContainer.appendChild(img);
            imgContainer.appendChild(location);
            resultsContainer.appendChild(imgContainer);

            const isAtBottom = resultsContainer.scrollHeight - resultsContainer.scrollTop - resultsContainer.clientHeight < 50;
            if (isAtBottom) {
                resultsContainer.scrollTop = resultsContainer.scrollHeight;
            }
        } catch (error) {
            console.error('Error processing message:', error);
            addErrorMessage(resultsContainer, 'Error processing detection result');
        }
    };

    eventSource.onerror = function(error) {
        console.error('SSE Error:', error);
        eventSource.close();
        addErrorMessage(resultsContainer, 'Detection process interrupted');
    };
}

function addErrorMessage(container, message) {
    const errorDiv = document.createElement('div');
    errorDiv.style.cssText = 'background-color: #ffebee; color: #c62828; padding: 10px; margin: 10px 0; border-radius: 4px;';
    errorDiv.textContent = message;
    container.appendChild(errorDiv);
}

// Initialize viewer when the script loads
document.addEventListener('DOMContentLoaded', function() {
    const metadata = window.slideMetadata;
    viewer = initializeViewer(
        metadata.slideName,
        metadata.height,
        metadata.width,
        metadata.tileSize,
        metadata.levels
    );
});