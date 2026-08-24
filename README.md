# workshare
sharing work stuffs

to increase performance of games
###gamemoderun RADV_PERFTEST=gpl %command% -USEALLAVAILABLECORES
##OR
###gamemoderun RADV_PERFTEST=gpl PROTON_LOCAL_SHADER_CACHE=1 %command%

script to run "AI"
```
import sounddevice as sd
from scipy.io.wavfile import write
import httpx
from google import genai
from google.genai import types

# Initialize Gemini client
client = genai.Client()

# Audio recording configurations
SAMPLE_RATE = 16000  # 16kHz is ideal for voice processing
DURATION = 5         # Recording duration in seconds (adjust or use a VAD tool for speech-end detection)

def record_from_jabra():
    print("\n🎤 Listening through your Jabra speaker... Speak now!")
    # Record audio from your default input device (the Jabra speakerphone)
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    
    input_filename = "user_input.wav"
    write(input_filename, SAMPLE_RATE, audio_data)
    print("Audio captured successfully.")
    return input_filename

def ask_gemini_multimodal(audio_path):
    print("Processing audio with Gemini (with Search Grounding)...")
    
    # Upload or pass the audio file directly to Gemini
    audio_file_ref = client.files.upload(file=audio_path)
    
    system_instruction = (
        "You are a personal travel assistant. "
        "The user lives in Tokyo, Japan, and their nearest station is JR Ueno Station. "
        "Whenever the user asks for directions, always calculate the route starting "
        "specifically from JR Ueno Station using current public transit data."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            audio_file_ref, 
            "Listen to this voice query and answer the question according to your system instructions."
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[{"google_search": {}}], # Enables live web lookups
        ),
    )
    
    print(f"\nGemini Answer: {response.text}\n")
    return response.text

def play_cloned_voice(text_response, ref_audio_path, ref_transcript):
    print("Synthesizing cloned voice via local vLLM Fish Audio...")
    
    payload = {
        "model": "fishaudio/s2-pro",
        "input": text_response,
        "voice": "default",
        "ref_audio": ref_audio_path,
        "ref_text": ref_transcript,
        "response_format": "wav"
    }

    tts_url = "http://localhost:8091/v1/audio/speech"
    output_filename = "response_output.wav"
    
    # Stream and save the synthesized cloned audio
    with httpx.stream("POST", tts_url, json=payload, timeout=300.0) as r:
        r.raise_for_status()
        with open(output_filename, "wb") as f:
            for data in r.iter_bytes():
                f.write(data)

    print("Playing response through Jabra speaker...")
    # Read the generated response and play it out of the speaker
    from scipy.io import wavfile
    samplerate, data = wavfile.read(output_filename)
    sd.play(data, samplerate)
    sd.wait()
    print("Done!")

if __name__ == "__main__":
    # Your reference voice configurations for zero-shot cloning
    MY_VOICE_SAMPLE = "./my_voice_clip.wav"
    MY_VOICE_TRANSCRIPT = "This is the exact sentence I spoke into my reference audio file."
    
    # Run the loop
    user_audio = record_from_jabra()
    gemini_text = ask_gemini_multimodal(user_audio)
    play_cloned_voice(gemini_text, MY_VOICE_SAMPLE, MY_VOICE_TRANSCRIPT)
```
