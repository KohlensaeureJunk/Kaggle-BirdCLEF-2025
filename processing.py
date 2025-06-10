import numpy as np
import librosa
import time, math
from tqdm.auto import tqdm
import cv2

"""
## Pre-processing
These functions handle the transformation of audio files to mel spectrograms for model input,
with flexibility controlled by the `LOAD_DATA` parameter. The process involves either loading
pre-computed spectrograms from this [dataset](https://www.kaggle.com/datasets/kadircandrisolu/birdclef25-mel-spectrograms)
(when `LOAD_DATA=True`) or dynamically generating them (when `LOAD_DATA=False`), transforming 
audio data into spectrogram representations, and preparing it for the neural network.
"""


def audio2melspec(audio_data, cfg):
    """Convert audio data to mel spectrogram"""
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm

def process_audio_file(audio_path, cfg, timestamp=None):
    """Process a single audio file to get the mel spectrogram
    
    Args:
        audio_path: Path to the audio file
        cfg: Configuration object
        timestamp: Optional timestamp (5, 10, 15, etc.) indicating which 5-second window to extract
    """
    try:
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
        target_samples = int(cfg.TARGET_DURATION * cfg.FS)

        if len(audio_data) < target_samples:
            n_copy = math.ceil(target_samples / len(audio_data))
            if n_copy > 1:
                audio_data = np.concatenate([audio_data] * n_copy)

        # Extract audio segment based on timestamp
        if timestamp is not None:
            # Calculate start time in seconds (timestamp represents the end of the 5-second window)
            start_time_seconds = max(0, timestamp - cfg.TARGET_DURATION)
            start_idx = int(start_time_seconds * cfg.FS)
            end_idx = min(len(audio_data), start_idx + target_samples)
            
            # Extract the specific 5-second window
            windowed_audio = audio_data[start_idx:end_idx]
            
            # Pad if necessary (shouldn't happen with proper timestamps but safety check)
            if len(windowed_audio) < target_samples:
                windowed_audio = np.pad(windowed_audio, 
                                      (0, target_samples - len(windowed_audio)), 
                                      mode='constant')
            audio_segment = windowed_audio
        else:
            # Extract center 5 seconds (original behavior for regular training data)
            start_idx = max(0, int(len(audio_data) / 2 - target_samples / 2))
            end_idx = min(len(audio_data), start_idx + target_samples)
            center_audio = audio_data[start_idx:end_idx]

            if len(center_audio) < target_samples:
                center_audio = np.pad(center_audio, 
                                     (0, target_samples - len(center_audio)), 
                                     mode='constant')
            audio_segment = center_audio

        mel_spec = audio2melspec(audio_segment, cfg)
        
        if mel_spec.shape != cfg.TARGET_SHAPE:
            mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

        return mel_spec.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def generate_spectrograms(df, cfg):
    """Generate spectrograms from audio files"""
    print("Generating mel spectrograms from audio files...")
    start_time = time.time()

    all_bird_data = {}
    errors = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        
        try:
            samplename = row['samplename']
            filepath = row['filepath']
            timestamp = row.get('timestamp', None)  # Get timestamp if available
            
            mel_spec = process_audio_file(filepath, cfg, timestamp=timestamp)
            
            if mel_spec is not None:
                all_bird_data[samplename] = mel_spec
            
        except Exception as e:
            print(f"Error processing {row.filepath}: {e}")
            errors.append((row.filepath, str(e)))

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    print(f"Successfully processed {len(all_bird_data)} files out of {len(df)}")
    print(f"Failed to process {len(errors)} files")
    
    return all_bird_data