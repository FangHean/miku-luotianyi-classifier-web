"use strict";

const MODEL_PART_URLS = [
  "./model/miku_luotianyi_resnet18.onnx.part-00",
  "./model/miku_luotianyi_resnet18.onnx.part-01",
];
const IMAGE_SIZE = 224;
const RESIZE_SIZE = 256;
const LUO_THRESHOLD = 0.65;
const MAX_FILE_BYTES = 20 * 1024 * 1024;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

const elements = {
  modelStatus: document.querySelector("#modelStatus"),
  statusText: document.querySelector("#statusText"),
  dropZone: document.querySelector("#dropZone"),
  fileInput: document.querySelector("#fileInput"),
  errorMessage: document.querySelector("#errorMessage"),
  result: document.querySelector("#result"),
  preview: document.querySelector("#preview"),
  previewBadge: document.querySelector("#previewBadge"),
  predictionName: document.querySelector("#predictionName"),
  confidenceChip: document.querySelector("#confidenceChip"),
  resultNote: document.querySelector("#resultNote"),
  mikuValue: document.querySelector("#mikuValue"),
  luoValue: document.querySelector("#luoValue"),
  mikuProgress: document.querySelector("#mikuProgress"),
  luoProgress: document.querySelector("#luoProgress"),
  mikuBar: document.querySelector("#mikuBar"),
  luoBar: document.querySelector("#luoBar"),
  tryAgain: document.querySelector("#tryAgain"),
  canvas: document.querySelector("#processingCanvas"),
};

let session = null;
let previewUrl = null;

function setStatus(message, state = "loading") {
  elements.statusText.textContent = message;
  elements.modelStatus.className = `model-status ${state}`;
}

function showError(message) {
  elements.errorMessage.textContent = message;
  elements.errorMessage.hidden = false;
}

function clearError() {
  elements.errorMessage.hidden = true;
  elements.errorMessage.textContent = "";
}

function formatPercent(value) {
  return new Intl.NumberFormat("zh-CN", {
    style: "percent",
    maximumFractionDigits: 1,
  }).format(value);
}

async function fetchPartWithProgress(url, partIndex, partCount) {
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok) {
    throw new Error(`模型下载失败（${response.status}）`);
  }

  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !total) {
    return response.arrayBuffer();
  }

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    setStatus(
      `模型下载中 · 第 ${partIndex + 1}/${partCount} 部分 ` +
        `${Math.round((received / total) * 100)}%`,
      "loading",
    );
  }

  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes.buffer;
}

async function fetchModel() {
  const partBuffers = [];
  for (let index = 0; index < MODEL_PART_URLS.length; index += 1) {
    partBuffers.push(
      await fetchPartWithProgress(
        MODEL_PART_URLS[index],
        index,
        MODEL_PART_URLS.length,
      ),
    );
  }

  const totalBytes = partBuffers.reduce(
    (total, buffer) => total + buffer.byteLength,
    0,
  );
  const modelBytes = new Uint8Array(totalBytes);
  let offset = 0;
  for (const buffer of partBuffers) {
    modelBytes.set(new Uint8Array(buffer), offset);
    offset += buffer.byteLength;
  }
  return modelBytes.buffer;
}

async function loadModel() {
  try {
    if (!globalThis.ort) {
      throw new Error("浏览器运行组件未能加载");
    }
    setStatus("正在准备模型…", "loading");
    ort.env.wasm.wasmPaths = new URL("./vendor/", window.location.href).href;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;

    const modelBuffer = await fetchModel();
    setStatus("正在初始化模型…", "loading");
    session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    elements.fileInput.disabled = false;
    elements.dropZone.classList.remove("disabled");
    setStatus("模型准备就绪", "ready");
  } catch (error) {
    console.error(error);
    setStatus("模型加载失败", "error");
    showError("模型未能加载。请刷新页面；如仍失败，请换用最新版 Chrome、Edge、Safari 或 Firefox。");
  }
}

function validateFile(file) {
  if (!file) return "没有选择图片。";
  if (file.size > MAX_FILE_BYTES) return "图片不能超过 20 MB。";
  const allowed = new Set([
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/gif",
  ]);
  if (!allowed.has(file.type)) return "请选择 JPG、PNG、WebP、BMP 或 GIF 图片。";
  return null;
}

async function decodeImage(file) {
  if ("createImageBitmap" in window) {
    try {
      return await createImageBitmap(file, { imageOrientation: "from-image" });
    } catch {
      return createImageBitmap(file);
    }
  }

  const objectUrl = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = objectUrl;
    await image.decode();
    return image;
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

function preprocess(source) {
  const width = source.naturalWidth || source.width;
  const height = source.naturalHeight || source.height;
  const scale = RESIZE_SIZE / Math.min(width, height);
  const cropWidth = IMAGE_SIZE / scale;
  const cropHeight = IMAGE_SIZE / scale;
  const sourceX = (width - cropWidth) / 2;
  const sourceY = (height - cropHeight) / 2;

  const context = elements.canvas.getContext("2d", {
    alpha: false,
    willReadFrequently: true,
  });
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.clearRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  context.drawImage(
    source,
    sourceX,
    sourceY,
    cropWidth,
    cropHeight,
    0,
    0,
    IMAGE_SIZE,
    IMAGE_SIZE,
  );

  const rgba = context.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;
  const planeSize = IMAGE_SIZE * IMAGE_SIZE;
  const tensorData = new Float32Array(3 * planeSize);

  for (let pixel = 0; pixel < planeSize; pixel += 1) {
    const rgbaOffset = pixel * 4;
    tensorData[pixel] = (rgba[rgbaOffset] / 255 - MEAN[0]) / STD[0];
    tensorData[planeSize + pixel] =
      (rgba[rgbaOffset + 1] / 255 - MEAN[1]) / STD[1];
    tensorData[2 * planeSize + pixel] =
      (rgba[rgbaOffset + 2] / 255 - MEAN[2]) / STD[2];
  }

  return new ort.Tensor("float32", tensorData, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
}

function showResult(mikuProbability, luoProbability, elapsedMs) {
  const isLuo = luoProbability >= LUO_THRESHOLD;
  const prediction = isLuo ? "洛天依" : "初音未来";
  const predictedProbability = isLuo ? luoProbability : mikuProbability;
  const needsReview = Math.max(mikuProbability, luoProbability) < 0.8;

  elements.predictionName.textContent = prediction;
  elements.result.classList.toggle("luo", isLuo);
  elements.confidenceChip.textContent = needsReview ? "建议复核" : "较高置信度";
  elements.confidenceChip.classList.toggle("review", needsReview);
  elements.previewBadge.textContent = `用时 ${Math.max(1, Math.round(elapsedMs))} ms`;

  if (needsReview) {
    elements.resultNote.textContent =
      "两类概率比较接近。建议换用主体更清晰、遮挡更少的单人图片再次判断。";
  } else {
    elements.resultNote.textContent =
      `模型给“${prediction}”的概率为 ${formatPercent(predictedProbability)}。` +
      "本结果仅供图片分类参考。";
  }

  const mikuPercent = Math.max(0, Math.min(100, mikuProbability * 100));
  const luoPercent = Math.max(0, Math.min(100, luoProbability * 100));
  elements.mikuValue.textContent = formatPercent(mikuProbability);
  elements.luoValue.textContent = formatPercent(luoProbability);
  elements.mikuProgress.setAttribute("aria-valuenow", mikuPercent.toFixed(1));
  elements.luoProgress.setAttribute("aria-valuenow", luoPercent.toFixed(1));

  elements.result.hidden = false;
  elements.dropZone.hidden = true;
  requestAnimationFrame(() => {
    elements.mikuBar.style.width = `${mikuPercent}%`;
    elements.luoBar.style.width = `${luoPercent}%`;
  });
}

async function classify(file) {
  const validationError = validateFile(file);
  if (validationError) {
    showError(validationError);
    return;
  }
  if (!session) {
    showError("模型仍在加载，请稍候再试。");
    return;
  }

  clearError();
  elements.fileInput.disabled = true;
  setStatus("正在识别…", "loading");

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  elements.preview.src = previewUrl;

  let decoded = null;
  try {
    decoded = await decodeImage(file);
    const startedAt = performance.now();
    const tensor = preprocess(decoded);
    const feeds = { [session.inputNames[0]]: tensor };
    const outputMap = await session.run(feeds);
    const probabilities = outputMap[session.outputNames[0]].data;
    showResult(Number(probabilities[0]), Number(probabilities[1]), performance.now() - startedAt);
    setStatus("识别完成", "ready");
  } catch (error) {
    console.error(error);
    showError("这张图片无法识别，请尝试转换为 JPG 或 PNG 后重试。");
    setStatus("模型准备就绪", "ready");
  } finally {
    if (decoded && typeof decoded.close === "function") decoded.close();
    elements.fileInput.disabled = false;
  }
}

elements.fileInput.addEventListener("change", (event) => {
  classify(event.target.files[0]);
});

for (const eventName of ["dragenter", "dragover"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    if (!elements.fileInput.disabled) elements.dropZone.classList.add("dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("dragging");
  });
}

elements.dropZone.addEventListener("drop", (event) => {
  if (!elements.fileInput.disabled) classify(event.dataTransfer.files[0]);
});

elements.tryAgain.addEventListener("click", () => {
  clearError();
  elements.result.hidden = true;
  elements.dropZone.hidden = false;
  elements.fileInput.value = "";
  elements.mikuBar.style.width = "0";
  elements.luoBar.style.width = "0";
  elements.dropZone.scrollIntoView({ behavior: "smooth", block: "center" });
});

elements.dropZone.classList.add("disabled");
setStatus("正在加载模型…", "loading");
loadModel();
