import spotdl
import librosa
import os
import matplotlib.pyplot as plt
import numpy as np

for filename in os.listdir("./song_db"):
    if os.path.isfile(os.path.join("./song_db", filename)):
        y, sr = librosa.load(os.path.join("./song_db", filename))
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram, sr=sr)
        plt.tight_layout()
        plt.savefig(os.path.join("./song_db/song_spectrograms", os.path.splitext(filename)[0] + ".jpg"))
        plt.close()

