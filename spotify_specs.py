import spotdl
import librosa
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tinytag
import argparse
import re

def create_specs(args):
    mode = args.mode
    if mode == "Train":
        filename_metadata = "./train_db/fma_metadata/tracks.csv"
        tracks = pd.read_csv(filename_metadata, header=2, low_memory=False)
        tracks_array = tracks.values
        tracks_id_array = tracks_array[:, 0]
        tracks_genre_array = tracks_array[:, 40]
        tracks_id_array = tracks_id_array.reshape(tracks_id_array.shape[0], 1)
        tracks_genre_array = tracks_genre_array.reshape(tracks_genre_array.shape[0], 1)

        train_folder = "./train_db/fma_small"
        directories = [d for d in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, d))]

        counter = 0
        for d in directories:
            label_directory = os.path.join(train_folder, d)
            filenames = [os.path.join(label_directory, f) for f in os.listdir(label_directory) if f.endswith(".mp3")]
            
            for filename in filenames:
                track_id = int(re.search(r'fma/.*/(.+?).mp3', filename).group(1))
                track_index = list(tracks_id_array).index(track_id)
                if (str(tracks_genre_array[track_index, 0]) != '0'):
                    y, sr = librosa.load(filename)
                    melspectrogram_array = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
                    spectrogram = librosa.power_to_db(melspectrogram_array)

                    fig_size = plt.rcParams["figure.figsize"]
                    fig_size[0] = float(spectrogram.shape[0]/100)
                    fig_size[1] = float(spectrogram.shape[1]/100)
                    plt.rcParams["figure.figsize"] = fig_size
                    plt.axis("off")
                    plt.figure(figsize=(10, 4))
                    plt.axes([0., 0., 1., 1.0], frameon=False, xticks=[], yticks=[])
                    librosa.display.specshow(spectrogram, cmap='grey_r')
                    plt.savefig("./train_db/train_spectrograms/"+ str(counter) + "_" + str(tracks_genre_array[track_index,0]) + ".jpg", dpi=100)
                    plt.close()
                    counter = counter + 1
        return
    elif mode == "Test":
        for filename in os.listdir("./song_db"):
            if os.path.isfile(os.path.join("./song_db", filename)):
                y, sr = librosa.load(os.path.join("./song_db", filename))
                melspectrogram_array = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
                spectrogram = librosa.power_to_db(melspectrogram_array)

                fig_size = plt.rcParams["figure.figsize"]
                fig_size[0] = float(spectrogram.shape[0]/100)
                fig_size[1] = float(spectrogram.shape[1]/100)
                plt.rcParams["figure.figsize"] = fig_size
                plt.axis("off")
                plt.figure(figsize=(10, 4))
                plt.axes([0., 0., 1., 1.0], frameon=False, xticks=[], yticks=[])
                librosa.display.specshow(spectrogram, cmap='grey_r')
                plt.savefig(os.path.join("./song_db/song_spectrograms", os.path.splitext(filename)[0] + ".jpg"))
                plt.close()
        return

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Convert mp3s to spectrograms")
    argparser.add_argument("-m", "--mode", required=True, help='set mode to create spectrograms for Train or Test')
    argparser.add_argument("-s", "--song", required=False, help='create spectrogram for given song')
    args = argparser.parse_args()
    create_specs(args)
