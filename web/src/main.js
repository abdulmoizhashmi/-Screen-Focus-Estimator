import "./style.css";

const VISION_VERSION = "1.2.2";
const LANDMARK_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VISION_VERSION}/wasm`;
const LANDMARK_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const IMAGE_SIZE = 160;
const INFERENCE_INTERVAL_MS = 180;
const EYE_LANDMARKS = [
  [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
  [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
];

const video = document.querySelector("#video");
const overlay = document.querySelector("#overlay");
const placeholder = document.querySelector("#placeholder");
const status = document.querySelector("#status");
const prediction = document.querySelector("#prediction");
const confidence = document.querySelector("#confidence");
const startButton = document.querySelector("#start");
const stopButton = document.querySelector("#stop");
const cropCanvas = document.createElement("canvas");
cropCanvas.width = IMAGE_SIZE;
cropCanvas.height = IMAGE_SIZE;
const cropContext = cropCanvas.getContext("2d", { willReadFrequently: true });
const overlayContext = overlay.getContext("2d");

let model;
let faceLandmarker;
let classNames;
let tf;
let FaceLandmarker;
let FilesetResolver;
let stream;
let animationFrame = 0;
let lastInference = 0;
let busy = false;

function setStatus(message) {
  status.textContent = message;
}

async function loadModels() {
  if (!tf || !FaceLandmarker || !FilesetResolver) {
    const [tfModule, visionModule] = await Promise.all([
      import("@tensorflow/tfjs"),
      import("@mediapipe/tasks-vision"),
    ]);
    tf = tfModule;
    ({ FaceLandmarker, FilesetResolver } = visionModule);
  }

  if (!model) {
    setStatus("Loading the classifier…");
    await tf.ready();
    model = await tf.loadGraphModel(`${import.meta.env.BASE_URL}model/model.json`);
    const labelsResponse = await fetch(
      `${import.meta.env.BASE_URL}model/class_names.json`,
    );
    if (!labelsResponse.ok) {
      throw new Error("Could not load the classifier labels.");
    }
    classNames = await labelsResponse.json();
    if (!Array.isArray(classNames) || classNames.length !== model.outputs.length) {
      throw new Error("Classifier labels do not match the model output.");
    }
    const warmup = tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
    const output = model.predict(warmup);
    tf.dispose([warmup, output]);
  }

  if (!faceLandmarker) {
    setStatus("Loading face landmark model…");
    const vision = await FilesetResolver.forVisionTasks(LANDMARK_WASM);
    const options = {
      baseOptions: { modelAssetPath: LANDMARK_MODEL, delegate: "GPU" },
      runningMode: "VIDEO",
      numFaces: 1,
    };
    try {
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, options);
    } catch (error) {
      console.warn("GPU face landmark delegate unavailable; using CPU.", error);
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        ...options,
        baseOptions: { modelAssetPath: LANDMARK_MODEL, delegate: "CPU" },
      });
    }
  }
}

function eyeBounds(landmarks, indices) {
  const points = indices.map((index) => landmarks[index]);
  const xs = points.map((point) => point.x * video.videoWidth);
  const ys = points.map((point) => point.y * video.videoHeight);
  const left = Math.min(...xs);
  const right = Math.max(...xs);
  const top = Math.min(...ys);
  const bottom = Math.max(...ys);
  const width = right - left;
  const height = bottom - top;
  const horizontalMargin = width * 0.30;
  const verticalMargin = height * 0.55;

  return {
    x: Math.max(0, left - horizontalMargin),
    y: Math.max(0, top - verticalMargin),
    width: Math.min(video.videoWidth, right + horizontalMargin) - Math.max(0, left - horizontalMargin),
    height: Math.min(video.videoHeight, bottom + verticalMargin) - Math.max(0, top - verticalMargin),
  };
}

async function classifyEyes(landmarks) {
  const predictions = [];

  for (const indices of EYE_LANDMARKS) {
    const bounds = eyeBounds(landmarks, indices);
    if (bounds.width < 12 || bounds.height < 8) continue;

    cropContext.drawImage(
      video,
      bounds.x,
      bounds.y,
      bounds.width,
      bounds.height,
      0,
      0,
      IMAGE_SIZE,
      IMAGE_SIZE,
    );
    const input = tf.browser.fromPixels(cropCanvas).toFloat().expandDims(0);
    const output = model.predict(input);
    predictions.push(Array.from(await output.data()));
    tf.dispose([input, output]);
  }

  if (!predictions.length) return null;
  return predictions[0].map(
    (_, classIndex) =>
      predictions.reduce((sum, values) => sum + values[classIndex], 0) /
      predictions.length,
  );
}

function drawEyeBounds(landmarks) {
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  overlayContext.strokeStyle = "#6de0cf";
  overlayContext.lineWidth = Math.max(2, overlay.width / 400);

  for (const indices of EYE_LANDMARKS) {
    const bounds = eyeBounds(landmarks, indices);
    overlayContext.strokeRect(bounds.x, bounds.y, bounds.width, bounds.height);
  }
}

async function inferFrame() {
  const now = performance.now();
  if (
    !stream ||
    busy ||
    video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
    now - lastInference < INFERENCE_INTERVAL_MS
  ) {
    return;
  }

  lastInference = now;
  busy = true;
  try {
    const result = faceLandmarker.detectForVideo(video, now);
    const landmarks = result.faceLandmarks[0];
    if (!landmarks) {
      overlayContext.clearRect(0, 0, overlay.width, overlay.height);
      prediction.textContent = "No face detected";
      confidence.textContent = "";
      setStatus("Center your face in the camera");
      return;
    }

    drawEyeBounds(landmarks);
    const probabilities = await classifyEyes(landmarks);
    if (!probabilities) {
      prediction.textContent = "Eyes not detected";
      confidence.textContent = "";
      setStatus("Adjust lighting and face the camera");
      return;
    }

    let bestIndex = 0;
    probabilities.forEach((value, index) => {
      if (value > probabilities[bestIndex]) bestIndex = index;
    });
    prediction.textContent = classNames[bestIndex].replaceAll("_", " ");
    confidence.textContent = `Model score: ${(probabilities[bestIndex] * 100).toFixed(1)}%`;
    setStatus("Camera is on — processing stays in this browser");
  } catch (error) {
    console.error("Frame inference failed:", error);
    stopCamera();
    setStatus("Inference failed. Reload the page and try again.");
  } finally {
    busy = false;
  }
}

function renderFrame() {
  if (!stream) return;
  void inferFrame();
  animationFrame = requestAnimationFrame(renderFrame);
}

async function startCamera() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    setStatus("Camera access requires HTTPS and a supported browser");
    return;
  }

  startButton.disabled = true;
  try {
    await loadModels();
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
    video.srcObject = stream;
    await video.play();
    placeholder.hidden = true;
    stopButton.disabled = false;
    setStatus("Camera is on — processing stays in this browser");
    renderFrame();
  } catch (error) {
    console.error("Could not start camera or load models:", error);
    stopCamera();
    startButton.disabled = false;
    if (error.name === "NotAllowedError") {
      setStatus("Camera permission was denied. Allow access in your browser settings.");
    } else if (error.name === "NotFoundError") {
      setStatus("No camera was found on this device.");
    } else {
      setStatus(`Could not start: ${error.message}`);
    }
  }
}

function stopCamera() {
  cancelAnimationFrame(animationFrame);
  stream?.getTracks().forEach((track) => track.stop());
  stream = undefined;
  video.srcObject = null;
  placeholder.hidden = false;
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  prediction.textContent = "—";
  confidence.textContent = "";
  stopButton.disabled = true;
  startButton.disabled = false;
  setStatus("Camera stopped");
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
window.addEventListener("pagehide", stopCamera);
