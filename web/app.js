const $ = (id) => document.getElementById(id);
const dropzone = $("dropzone");
const fileInput = $("file");
const preview = $("preview");
const runButton = $("run");
const status = $("status");

let selected = null; // {file} for an upload, or {sample} for a bundled image
let models = [];

const checked = (name) => document.querySelector(`input[name="${name}"]:checked`).value;
const currentModel = () =>
  models.find((m) => m.decoder === checked("decoder") && m.training === checked("training"));

function setStatus(text, isError = false) {
  status.textContent = text;
  status.classList.toggle("error", isError);
}

function showImage(src) {
  preview.src = src;
  preview.hidden = false;
  dropzone.classList.add("has-image");
  runButton.disabled = false;
  setStatus("");
}

function pickFile(file) {
  if (!file || !file.type.startsWith("image/")) return setStatus("That file is not an image.", true);
  selected = { file };
  showImage(URL.createObjectURL(file));
}

/* ---------- choosing an image ---------- */
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", (e) => pickFile(e.target.files[0]));

["dragenter", "dragover"].forEach((type) =>
  dropzone.addEventListener(type, (e) => { e.preventDefault(); dropzone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((type) =>
  dropzone.addEventListener(type, (e) => { e.preventDefault(); dropzone.classList.remove("dragging"); }));
dropzone.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));

document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) pickFile(item.getAsFile());
});

/* ---------- controls ---------- */
function syncBeamInputs() {
  const manual = checked("strategy") === "beam" && !$("tuned").checked;
  $("beam-size").disabled = !manual;
  $("length-penalty").disabled = !manual;
  const model = currentModel();
  if (model && $("tuned").checked) {
    $("beam-size").value = model.beam_size;
    $("length-penalty").value = String(model.length_penalty);
  }
  $("beam-fields").style.opacity = checked("strategy") === "beam" ? "1" : ".5";
}

document.querySelectorAll('input[name="strategy"], input[name="decoder"], input[name="training"]')
  .forEach((el) => el.addEventListener("change", syncBeamInputs));
$("tuned").addEventListener("change", syncBeamInputs);

/* ---------- captioning ---------- */
async function generate() {
  if (!selected) return;
  const body = new FormData();
  if (selected.file) body.append("file", selected.file);
  else body.append("sample_name", selected.sample);
  body.append("decoder", checked("decoder"));
  body.append("training", checked("training"));
  body.append("strategy", checked("strategy"));
  body.append("beam_size", $("beam-size").value);
  body.append("length_penalty", $("length-penalty").value);

  runButton.disabled = true;
  setStatus("Captioning…");
  $("result").classList.add("loading");
  $("attention-block").classList.add("loading");

  try {
    const response = await fetch("/api/caption", { method: "POST", body });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    render(await response.json());
    setStatus("");
  } catch (error) {
    setStatus(`Could not caption that image: ${error.message}`, true);
  } finally {
    runButton.disabled = false;
    $("result").classList.remove("loading");
    $("attention-block").classList.remove("loading");
  }
}

function render(data) {
  $("placeholder").hidden = true;
  $("result").hidden = false;
  $("caption").textContent = data.caption;

  const score = data.test_cider === null ? "" :
    `<span class="dot">·</span><span class="score">test CIDEr ${data.test_cider}</span>`;
  $("meta").innerHTML =
    `${data.model}<span class="dot">·</span>${data.setting}<span class="dot">·</span>${data.words} words${score}`;

  $("attention-block").hidden = false;
  $("tiles").innerHTML = data.tiles.map((tile) => `
    <figure class="tile${tile.word === "<eos>" ? " eos" : ""}">
      <img src="${tile.image}" alt="Attention while generating “${tile.word}”">
      <span>${tile.word === "<eos>" ? "end" : tile.word}</span>
    </figure>`).join("");
}

runButton.addEventListener("click", generate);

/* ---------- start-up ---------- */
(async function init() {
  try {
    const info = await (await fetch("/api/models")).json();
    models = info.models;
    $("device").textContent = `Running on ${info.device.toUpperCase()}.`;

    // Disable model options that have no checkpoint on disk.
    document.querySelectorAll('input[name="decoder"], input[name="training"]').forEach((input) => {
      const key = input.name === "decoder" ? "decoder" : "training";
      if (!models.some((m) => m[key] === input.value)) input.disabled = true;
    });

    $("samples").innerHTML = info.samples.map((name) => `
      <button type="button" data-sample="${name}" title="Use this sample image">
        <img src="/api/sample/${name}" alt="">
      </button>`).join("");
    $("samples").querySelectorAll("button").forEach((button) =>
      button.addEventListener("click", () => {
        selected = { sample: button.dataset.sample };
        showImage(`/api/sample/${button.dataset.sample}`);
        generate();
      }));
    syncBeamInputs();

    // /?sample=<file> captions that sample straight away, so a link can show a finished result.
    const requested = new URLSearchParams(location.search).get("sample");
    if (requested && info.samples.includes(requested)) {
      selected = { sample: requested };
      showImage(`/api/sample/${requested}`);
      generate();
    }
  } catch {
    setStatus("Could not reach the server.", true);
  }
})();
