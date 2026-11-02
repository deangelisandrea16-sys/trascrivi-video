import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

env.allowLocalModels = false;

let asr = null, current = null, device = 'wasm';

async function hasWebGPU() {
  try { return !!(navigator.gpu && await navigator.gpu.requestAdapter()); } catch { return false; }
}

async function load(model) {
  if (asr && current === model) return asr;
  device = (await hasWebGPU()) ? 'webgpu' : 'wasm';
  const dtype = device === 'webgpu'
    ? { encoder_model: 'fp32', decoder_model_merged: 'q4' }
    : 'q8';
  asr = await pipeline('automatic-speech-recognition', model, {
    device, dtype,
    progress_callback: p => {
      if (p.status === 'progress') self.postMessage({ type: 'loading', file: p.file, progress: p.progress });
    }
  });
  current = model;
  return asr;
}

self.onmessage = async ({ data }) => {
  try {
    const t = await load(data.model);
    self.postMessage({ type: 'ready', device });
    const opts = { task: 'transcribe', chunk_length_s: 30, stride_length_s: 5, return_timestamps: data.timestamps };
    if (data.language) opts.language = data.language;
    const out = await t(data.audio, opts);
    self.postMessage({ type: 'done', text: out.text || '', chunks: out.chunks || [] });
  } catch (e) {
    self.postMessage({ type: 'error', message: e?.message || String(e) });
  }
};
