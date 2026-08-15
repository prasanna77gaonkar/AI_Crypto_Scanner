const API_URL = "https://ai-crypto-scanner-nvrv.onrender.com/api/chart/analyze";

const camera = document.getElementById("camera");
const preview = document.getElementById("imagePreview");
const previewHint = document.getElementById("previewHint");
const startCameraButton = document.getElementById("startCamera");
const captureButton = document.getElementById("captureChart");
const uploadInput = document.getElementById("imageUpload");
const scanButton = document.getElementById("scanChart");
const captureStatus = document.getElementById("captureStatus");
const emptyResult = document.getElementById("emptyResult");
const resultContent = document.getElementById("resultContent");
const resultState = document.getElementById("resultState");
const signal = document.getElementById("signal");
const confidence = document.getElementById("confidence");
const trend = document.getElementById("trend");
const timeframe = document.getElementById("timeframe");
const analysis = document.getElementById("analysis");

let cameraStream = null;
let selectedImage = null;
let previewUrl = null;

function clearPreviewUrl() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
}

function setSelectedImage(file) {
    selectedImage = file;
    clearPreviewUrl();
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.hidden = false;
    camera.hidden = true;
    previewHint.hidden = true;
    scanButton.disabled = false;
    captureStatus.textContent = "Chart image ready to scan.";
}

function showResult({ title, kind, confidenceText = "", trendText = "", timeframeText = "", analysisText }) {
    emptyResult.hidden = true;
    resultContent.hidden = false;
    signal.textContent = title;
    signal.className = `signal ${kind}`;
    confidence.textContent = confidenceText;
    trend.textContent = trendText;
    timeframe.textContent = timeframeText;
    analysis.textContent = analysisText;
}

async function startCamera() {
    try {
        if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: { ideal: "environment" } },
            audio: false
        });
        camera.srcObject = cameraStream;
        camera.hidden = false;
        preview.hidden = true;
        previewHint.hidden = true;
        captureButton.disabled = false;
        captureStatus.textContent = "Camera is ready. Point it at a candlestick chart.";
    } catch (error) {
        captureStatus.textContent = `Camera unavailable: ${error.message}`;
    }
}

function captureChart() {
    if (!cameraStream || !camera.videoWidth || !camera.videoHeight) {
        captureStatus.textContent = "The camera frame is not ready yet.";
        return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = camera.videoWidth;
    canvas.height = camera.videoHeight;
    canvas.getContext("2d").drawImage(camera, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
        if (!blob) {
            captureStatus.textContent = "Unable to capture the camera image.";
            return;
        }
        setSelectedImage(new File([blob], "captured-chart.jpg", { type: "image/jpeg" }));
    }, "image/jpeg", 0.9);
}

async function scanChart() {
    if (!selectedImage) return;
    scanButton.disabled = true;
    resultState.textContent = "Analyzing…";
    showResult({ title: "Scanning…", kind: "hold", analysisText: "The chart image is being analyzed by the scanner backend." });

    try {
        const formData = new FormData();
        formData.append("file", selectedImage);
        const response = await fetch(API_URL, { method: "POST", body: formData });
        let result;
        try {
            result = await response.json();
        } catch (_) {
            throw new Error("The scanner backend returned an invalid response.");
        }
        const scanner = result.scanner;
        if (!response.ok || !scanner) throw new Error(result.message || "The scanner backend could not analyze this image.");

        if (scanner.signal === "INVALID CHART") {
            resultState.textContent = "Invalid chart";
            showResult({
                title: "Invalid Chart",
                kind: "invalid",
                analysisText: "Please upload or capture a clear candlestick trading chart."
            });
            return;
        }

        const signalClass = scanner.signal.toLowerCase();
        resultState.textContent = "Analysis complete";
        showResult({
            title: scanner.signal,
            kind: signalClass,
            confidenceText: `${scanner.confidence}%`,
            trendText: scanner.trend,
            timeframeText: scanner.timeframe,
            analysisText: scanner.analysis
        });
    } catch (_) {
        resultState.textContent = "Connection failed";
        showResult({
            title: "Connection Failed",
            kind: "error",
            analysisText: "Unable to reach the scanner backend. Please check your internet connection and try again."
        });
    } finally {
        scanButton.disabled = false;
    }
}

startCameraButton.addEventListener("click", startCamera);
captureButton.addEventListener("click", captureChart);
uploadInput.addEventListener("change", (event) => {
    const [file] = event.target.files;
    if (file) setSelectedImage(file);
});
scanButton.addEventListener("click", scanChart);
window.addEventListener("beforeunload", () => {
    clearPreviewUrl();
    if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
});
