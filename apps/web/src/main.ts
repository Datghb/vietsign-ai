import { api, type ModelInfo, type PipelineInfo } from "./api";

const statusPill = element<HTMLSpanElement>("status-pill");
const pipelineList = element<HTMLDivElement>("pipeline-list");
const modelList = element<HTMLDivElement>("model-list");
const result = element<HTMLPreElement>("video-result");
const featuresPath = element<HTMLInputElement>("features-path");
const camera = element<HTMLVideoElement>("camera");
const videoFile = element<HTMLInputElement>("video-file");
const videoPreview = element<HTMLVideoElement>("video-preview");
let stream: MediaStream | null = null;

document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach((button) => {
  button.addEventListener("click", () => switchView(button.dataset.view ?? "home"));
});

element<HTMLButtonElement>("run-continuous").addEventListener("click", async () => {
  await runAction(() => api.recognizeContinuous(featuresPath.value));
});

element<HTMLButtonElement>("run-translate").addEventListener("click", async () => {
  await runAction(() => api.recognizeAndTranslate(featuresPath.value));
});

videoFile.addEventListener("change", () => {
  const file = videoFile.files?.[0];
  videoPreview.src = file ? URL.createObjectURL(file) : "";
});

element<HTMLButtonElement>("submit-video").addEventListener("click", () => {
  result.textContent = "Raw-video submission is available in the API shell, but real extraction/checkpoint readiness is deferred. Use feature-cache synthetic demos for now.";
});

element<HTMLButtonElement>("start-camera").addEventListener("click", async () => {
  stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  camera.srcObject = stream;
});

element<HTMLButtonElement>("stop-camera").addEventListener("click", () => {
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  camera.srcObject = null;
});

await refresh();

async function refresh(): Promise<void> {
  try {
    const [health, pipelines, models] = await Promise.all([api.health(), api.pipelines(), api.models()]);
    statusPill.textContent = String(health.status ?? "unknown");
    statusPill.className = `pill ${health.real_vsl_ready ? "ready" : "dev"}`;
    renderPipelines(pipelines.pipelines);
    renderModels(models.models);
  } catch (error) {
    statusPill.textContent = "API offline";
    statusPill.className = "pill offline";
    pipelineList.textContent = String(error);
  }
}

async function runAction(action: () => Promise<Record<string, unknown>>): Promise<void> {
  result.textContent = "Running...";
  try {
    result.textContent = JSON.stringify(await action(), null, 2);
  } catch (error) {
    result.textContent = String(error);
  }
}

function renderPipelines(pipelines: PipelineInfo[]): void {
  pipelineList.replaceChildren(...pipelines.map((item) => card(item.name, [
    `Task: ${item.task}`,
    `Ready: ${item.ready ? "yes" : "no"}`,
    item.reason ? `Reason: ${item.reason}` : "",
    item.development_only ? "Development only" : "",
  ])));
}

function renderModels(models: ModelInfo[]): void {
  modelList.replaceChildren(...models.map((item) => card(item.name, [
    `Task: ${item.task}`,
    `Modality: ${item.modality}`,
    `Checkpoint: ${item.checkpoint ?? "not configured"}`,
    `Loaded: ${item.loaded ? "yes" : "no"}`,
    item.development_only ? "Development only" : "",
    ...item.warnings,
  ])));
}

function card(title: string, lines: string[]): HTMLElement {
  const wrapper = document.createElement("article");
  wrapper.className = "card";
  const heading = document.createElement("h3");
  heading.textContent = title;
  wrapper.append(heading);
  lines.filter(Boolean).forEach((line) => {
    const paragraph = document.createElement("p");
    paragraph.textContent = line;
    wrapper.append(paragraph);
  });
  return wrapper;
}

function switchView(view: string): void {
  document.querySelectorAll(".view").forEach((section) => section.classList.toggle("active", section.id === view));
  document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
}

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) {
    throw new Error(`Missing element: ${id}`);
  }
  return found as T;
}
