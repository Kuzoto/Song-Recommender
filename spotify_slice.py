import spotdl
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import tinytag
import argparse
import re
import warnings
warnings.filterwarnings('ignore')

def slice_specs(args):
    mode = args.mode
    if mode == "Train":
        spec_folder = "./train_db/train_spectrograms"
        filenames = [os.path.join(spec_folder, f) for f in os.listdir(spec_folder) if f.endswith(".jpg")]
        counter = 0

        for f in filenames:
            genre = re.search(r'./train_db/train_spectrograms\\.*_(.+?).jpg', f).group(1)
            img = Image.open(f)
            subsample_size = 128
            width, height = img.size
            sample_num = width / subsample_size
            for i in range(round(sample_num)):
                start = i*subsample_size
                temp_img = img.crop((start, 0., start + subsample_size, subsample_size))
                temp_img.save("./train_db/train_slices/"+str(counter)+"_"+genre+".jpg")
                counter += 1
        return
    elif mode == "Test":
        spec_folder = "./song_db/song_spectrograms"
        filenames = [os.path.join(spec_folder, f) for f in os.listdir(spec_folder) if f.endswith(".jpg")]
        counter = 0

        for f in filenames:
            song = re.search(r'./song_db/song_spectrograms\\(.+?).jpg', f).group(1)
            img = Image.open(f)
            subsample_size = 128
            sample_num = 26
            for i in range(6, sample_num):
                start = i*subsample_size
                temp_img = img.crop((start, 0., start + subsample_size, subsample_size))
                temp_img.save("./song_db/song_slices/"+str(counter)+"_"+song+".jpg")
                counter += 1
        return
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Slice the song spectrograms")
    argparser.add_argument("-m", "--mode", required=True, help='set mode to create spectrograms for Train or Test')
    args = argparser.parse_args()
    slice_specs(args)