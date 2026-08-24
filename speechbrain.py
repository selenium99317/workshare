from speechbrain.inference.speaker import EncoderClassifier
import torch
import torch.nn.functional as F

# Load the ECAPA-TDNN model
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def get_embedding(audio_np, sample_rate=16000):
    # Convert numpy audio buffer to torch tensor
    signal = torch.tensor(audio_np).float().unsqueeze(0)
    embedding = classifier.encode_batch(signal)
    # Normalize the embedding vector
    return F.normalize(embedding.squeeze(0), dim=-1)

# --- ENROLLMENT (Do this once for your profiles) ---
# david_audio_np = ... (record 3 seconds of David speaking)
# kelvin_audio_np = ... (record 3 seconds of Kelvin speaking)
# david_vector = get_embedding(david_audio_np)
# kelvin_vector = get_embedding(kelvin_audio_np)

# --- LIVE RECOGNITION (Inside your main loop) ---
def identify_speaker(live_audio_np, david_vector, kelvin_vector, threshold=0.75):
    live_vector = get_embedding(live_audio_np)
    
    # Calculate cosine similarity against enrolled speakers
    score_david = F.cosine_similarity(live_vector, david_vector, dim=-1).item()
    score_kelvin = F.cosine_similarity(live_vector, kelvin_vector, dim=-1).item()
    
    print(f"Scores -> David: {score_david:.2f}, Kelvin: {score_kelvin:.2f}")
    
    if score_david > threshold and score_david > score_kelvin:
        return "David"
    elif score_kelvin > threshold and score_kelvin > score_david:
        return "Kelvin"
    else:
        return "Unknown"
