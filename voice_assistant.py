#!/usr/bin/env python3
#install packages first in .venv
#using bosonai/higgs-audio-v3-tts-4b model
#optional : try https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb for voice recognition
#pip install google-genai sounddevice soundfile numpy faster-whisper webrtcvad vllm-omni transformers
import os
import time
import collections
import numpy as np
import sounddevice as sd
import torch
import webrtcvad

# Configure environment for AMD ROCm on RDNA3 (RX 7800 XT)
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")

from google import genai
from faster_whisper import WhisperModel
from vllm_omni import Omni
from transformers import AutoTokenizer
from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_tokenizer import HiggsAudioV3TokenizerAdapter

# Configuration parameters
SAMPLE_RATE = 16000 
OUT_SAMPLE_RATE = 24_000
FRAME_DURATION_MS = 30 
BLOCK_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

def select_audio_device(kind="input"):
    devices = sd.query_devices()
    valid_devices = []
    
    print(f"\nAvailable {kind} devices:")
    for idx, dev in enumerate(devices):
        if dev[kind + '_channels'] > 0:
            valid_devices.append((idx, dev['name']))
            print(f"  [{idx}] {dev['name']}")
            
    if not valid_devices:
        print(f"No {kind} devices found. Using system default.")
        return None

    choice = input(f"Select {kind} device ID (or press Enter for default): ").strip()
    if choice.isdigit() and int(choice) in [d[0] for d in valid_devices]:
        return int(choice)
    
    print("Using system default device.")
    return None

def _extract_pcm(multimodal_output: dict) -> torch.Tensor:
    audio = multimodal_output.get("model_outputs") or multimodal_output.get("audio")
    if audio is None:
        raise ValueError("No audio key found in multimodal_output")
    if isinstance(audio, list):
        valid = [torch.as_tensor(a).float().cpu().reshape(-1) for a in audio if a is not None]
        return torch.cat(valid, dim=0) if len(valid) > 1 else valid[0]
    return torch.as_tensor(audio).float().cpu().reshape(-1)

def _pcm_to_int16(pcm: torch.Tensor) -> np.ndarray:
    arr = pcm.numpy()
    if arr.dtype.kind == "f":
        arr = np.clip(arr, -1.0, 1.0)
        arr = (arr * 32767.0).astype(np.int16)
    else:
        arr = arr.astype(np.int16)
    return arr

def main():
    print("Initializing Gemini Client...")
    gemini_client = genai.Client()

    print("Loading local Whisper model...")
    stt_model = WhisperModel("base", device="cpu", compute_type="int8")

    print("Loading local vLLM-Omni Higgs Audio v3 model on RX 7800 XT...")
    model_id = "bosonai/higgs-audio-v3-tts-4b"
    engine = Omni(model=model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    adapter = HiggsAudioV3TokenizerAdapter(tokenizer)

    # Dynamic device selection for whatever headphones/phone/speaker you are using now
    print("\n--- Audio Device Setup ---")
    input_device_idx = select_audio_device(kind="input")
    output_device_idx = select_audio_device(kind="output")

    vad = webrtcvad.Vad(2) # Aggressiveness mode (0 to 3)

    print("\n--- Hands-Free Voice Assistant Active ---")
    print("Just start speaking. Press Ctrl+C to exit.\n")

    try:
        while True:
            print("Listening for speech...", end="\r", flush=True)
            
            triggered = False
            voiced_frames = collections.deque(maxlen=10) 
            ring_buffer = collections.deque(maxlen=15)     
            recorded_frames = []

            with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1, 
                                dtype="int16", device=input_device_idx) as stream:
                while True:
                    frame, overflow = stream.read(BLOCK_SIZE)
                    if overflow:
                        continue
                    
                    bytes_data = frame.tobytes()
                    is_speech = vad.is_speech(bytes_data, SAMPLE_RATE)

                    if not triggered:
                        voiced_frames.append((bytes_data, is_speech))
                        num_voiced = len([f for f, speech in voiced_frames if speech])
                        if num_voiced > 0.5 * voiced_frames.maxlen:
                            triggered = True
                            print("\n[Speech detected! Recording...]")
                            for f, _ in voiced_frames:
                                recorded_frames.append(f)
                            voiced_frames.clear()
                    else:
                        recorded_frames.append(bytes_data)
                        ring_buffer.append((bytes_data, is_speech))
                        
                        num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                        if ring_buffer.maxlen and num_unvoiced > 0.8 * ring_buffer.maxlen:
                            break 

            print("Processing voice input...")
            audio_buffer = b"".join(recorded_frames)
            audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

            segments, _ = stt_model.transcribe(audio_np, beam_size=1)
            user_text = " ".join([segment.text for segment in segments]).strip()

            if not user_text:
                print("Could not understand audio. Listening again...\n")
                continue

            print(f"You said: \"{user_text}\"")

            print("Asking Gemini Flash...")
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_text,
            )
            gemini_reply = response.text.strip()
            print(f"Gemini says: \"{gemini_reply}\"")

            print("Synthesizing voice response locally...")
            prompt_ids = adapter.build_prompt(gemini_reply)
            outputs = engine.generate([{"prompt_token_ids": prompt_ids}])

            mm = outputs[0].outputs[0].multimodal_output
            pcm = _extract_pcm(mm)
            wav_int16 = _pcm_to_int16(pcm)

            print("Playing response...")
            sd.play(wav_int16, samplerate=OUT_SAMPLE_RATE, device=output_device_idx)
            sd.wait()
            print("-" * 50 + "\n")

    except KeyboardInterrupt:
        print("\nExiting voice assistant. Goodbye!")

if __name__ == "__main__":
    main()
