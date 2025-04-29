import spotdl
import subprocess
import argparse
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import tinytag
import re
import warnings

argparser = argparse.ArgumentParser(description="Download a song to add to the song_db")
argparser.add_argument("-u", "--url", required=True, help="url of the song to add to the song_db")
args = argparser.parse_args()


result = subprocess.run(['spotdl', args.url], capture_output=True, text=True, cwd=".\\song_db")
song_name = re.search(r'(?<=\")(.*?)(?=\")', result.stdout).group(1)
song_name = song_name.split(" - ")
for f in os.listdir("./song_db"):
    if song_name[0] in f and song_name[1] in f:
        filename = f
# Check if the command was successful
if result.returncode == 0:
    if os.path.isfile(os.path.join("./song_db", filename)):
        y, sr = librosa.load(os.path.join("./song_db", filename))
        melspectrogram_array = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        spectrogram = librosa.power_to_db(melspectrogram_array)

        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = float(spectrogram.shape[1]/100)
        fig_size[1] = float(spectrogram.shape[0]/100)
        plt.rcParams["figure.figsize"] = fig_size
        plt.axis("off")
        # plt.figure(figsize=(10, 4))
        plt.axes([0., 0., 1., 1.0], frameon=False, xticks=[], yticks=[])
        librosa.display.specshow(spectrogram, cmap='grey_r')
        f = os.path.join("./song_db/song_spectrograms", os.path.splitext(filename)[0] + ".jpg")
        plt.savefig(f)
        plt.close()
        
        #Provides a visualization of the audio signal
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(range(len(y)), y)
        # ax.set_title("Audio Signal")
        # ax.set_ylabel("Amplitude")
        # ax.set_xlabel("Time")
        # plt.show()


        counter = len(os.listdir("./song_db/song_slices")) - 1
        song = re.search(r'./song_db/song_spectrograms\\(.+?).jpg', f).group(1)
        img = Image.open(f)
        subsample_size = 128
        sample_num = 26
        for i in range(6, sample_num):
            start = i*subsample_size
            temp_img = img.crop((start, 0., start + subsample_size, subsample_size))
            temp_img.save("./song_db/song_slices/"+str(counter)+"_"+song+".jpg")
            counter += 1