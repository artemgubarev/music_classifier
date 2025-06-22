import os
import math
import librosa
import numpy as np
import tempfile
import subprocess
import imageio_ffmpeg
from scipy.stats import mode

genres = {
    0: '🎸 Blues',
    1: '🎻 Classical',
    2: '🤠 Country',
    3: '🪩 Disco',
    4: '🎤 Hip-Hop',
    5: '🎷 Jazz',
    6: '🤘 Metal',
    7: '🎧 Pop',
    8: '🌴 Reggae',
    9: '🎶 Rock'
}

def convert_to_wav_ffmpeg(input_path):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav.close()
    try:
        subprocess.run([
            ffmpeg_path, "-y", "-i", input_path,
            "-ar", "44100", "-ac", "1",
            temp_wav.name
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return temp_wav.name
    except subprocess.CalledProcessError:
        os.unlink(temp_wav.name)
        raise RuntimeError(f"FFmpeg failed to convert file: {input_path}")

def extract_features(audio_path, n_mfcc=13, sr_target=22050, n_fft=2048, 
                     hop_length=512, duration=30, num_segments=10):
    temp_file = None
    try:
        try:
            signal, sr = librosa.load(audio_path, sr=sr_target)
        except Exception as e:
            print(f"[librosa.load failed]: {e}. Trying FFmpeg conversion...")
            temp_file = convert_to_wav_ffmpeg(audio_path)
            signal, sr = librosa.load(temp_file, sr=sr_target)

        samples_per_track = duration * sr_target
        samples_per_segment = int(samples_per_track / num_segments)
        expected_frames = math.ceil(samples_per_segment / hop_length)

        if len(signal) < samples_per_segment:
            return None  # слишком короткий трек

        segments_mfcc = []

        for s in range(num_segments):
            start = samples_per_segment * s
            finish = start + samples_per_segment

            if len(signal[start:finish]) < samples_per_segment:
                continue

            mfcc = librosa.feature.mfcc(
                y=signal[start:finish],
                sr=sr,
                n_fft=n_fft,
                n_mfcc=n_mfcc,
                hop_length=hop_length
            ).T

            if len(mfcc) == expected_frames:
                segments_mfcc.append(mfcc)

        return np.array(segments_mfcc) if segments_mfcc else None

    finally:
        if temp_file is not None and os.path.exists(temp_file):
            os.remove(temp_file)

def predict_genre(model, audio_path):
    features = extract_features(audio_path)
    
    if features is None or features.shape[0] == 0:
        return "❗️ Аудиофайл слишком короткий или повреждён. Отправьте, пожалуйста, другой файл."

    features = features[..., np.newaxis]
    predictions = model.predict(features)
    predicted_indices = np.argmax(predictions, axis=1)
    final_prediction = mode(predicted_indices, keepdims=True).mode[0]

    return f"🎶 Предсказанный жанр: {genres[final_prediction]}"

def get_audio_duration(audio_path):
    temp_file = None
    try:
        try:
            y, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            print(f"[librosa.load failed for duration]: {e}. Trying FFmpeg conversion...")
            temp_file = convert_to_wav_ffmpeg(audio_path)
            y, sr = librosa.load(temp_file, sr=None)

        return librosa.get_duration(y=y, sr=sr)

    finally:
        if temp_file is not None and os.path.exists(temp_file):
            os.remove(temp_file)