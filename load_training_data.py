import os
import re
import cv2
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

def load_data(args):
    trainingSize = args.size
    genre = {
        "Hip-Hop": 0,
        "International": 1,
        "Electronic": 2,
        "Folk": 3,
        "Experimental": 4,
        "Rock": 5,
        "Pop": 6,
        "Instrumental": 7
    }
    filenames = [os.path.join("./train_db/train_slices", f) for f in os.listdir("./train_db/train_slices") if f.endswith(".jpg")]
    images = [None]*(len(filenames))
    labels = [None]*(len(filenames))
    for f in filenames:
        index = int(re.search(r'./train_db/train_slices\\(.+?)_.*.jpg', f).group(1))
        slice_genre = re.search(r'./train_db/train_slices\\.*_(.+?).jpg', f).group(1)
        temp = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        images[index] = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        labels[index] = genre[slice_genre]
    if trainingSize == 1.0:
        training_images = images
        training_labels = labels
    else:
        count_max = int(len(images)*trainingSize / 8.0)
        count_array = [0, 0, 0, 0, 0, 0, 0, 0]
        training_images = []
        training_labels = []
        for i in range(0, len(images)):
            if(count_array[labels[i]] < count_max):
                training_images.append(images[i])
                training_labels.append(labels[i])
                count_array[labels[i]] += 1

    training_images = np.array(training_images)
    training_labels = np.array(training_labels)
    training_labels = training_labels.reshape(training_labels.shape[0], 1)
    train_x, test_x, train_y, test_y = train_test_split(training_images, training_labels, test_size=0.05, shuffle=True)
    n_classes = len(genre)
    train_genre = {value: key for key, value in genre.items()}
    np.save("./train_db/train_x.npy", train_x) 
    np.save("./train_db/test_x.npy", test_x)
    np.save("./train_db/train_y.npy", train_y)
    np.save("./train_db/test_y.npy", test_y)
    return train_x, train_y, test_x, test_y, n_classes, train_genre

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Load the training data")
    argparser.add_argument("-s", "--size", type=float, default="1.0", help='set the training split size')
    args = argparser.parse_args()
    load_data(args)