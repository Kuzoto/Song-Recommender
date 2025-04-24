import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import math
import torch.nn as nn
import torch.nn.functional as F 
from torchsummary import summary
# from logger import Logger
import copy
from model import model
import re
import cv2
import os
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainDataset(Dataset):
    def __init__(self):
        self.x = torch.from_numpy(np.load("./train_db/train_x.npy").astype(np.float32))
        self.y = torch.from_numpy(np.load("./train_db/train_y.npy").reshape(-1, 1).astype(np.float32))
        self.n_samples = self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

class TestDataset(Dataset):
    def __init__(self):
        self.x = torch.from_numpy(np.load("./train_db/test_x.npy").astype(np.float32))
        self.y = torch.from_numpy(np.load("./train_db/test_y.npy").reshape(-1, 1).astype(np.float32))
        self.n_samples = self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
    
class RecommendModel:
    def __init__(self, func):
        self.function = func

    def __call__(self, *args, **kwargs):
        model_val = model.to(device)

        model_val.load_state_dict(torch.load('model.pt', map_location='cpu'))
        model_val.eval()

        removed = list(model.children())[:-1]
        rec_model = nn.Sequential(*removed)

        images, labels = self.function(*args, **kwargs)
        images = images[:, None, :, :]
        print(images.shape)
        images = images / 255.

        LIST_SONG = [('0', ' ')]
        for k, v in enumerate(np.unique(labels)):
            LIST_SONG.append(('{}'.format(k+1, v)))
        
        print(LIST_SONG)
        return LIST_SONG, images, labels, rec_model

@RecommendModel
def load_data():
    filenames = [os.path.join("./song_db/song_slices", f) for f in os.listdir("./song_db/song_slices") if f.endswith(".jpg")]

    images = []
    labels = []
    for f in filenames:
        song = re.search(r'./song_db/song_slices\\.*_(.+?).jpg', f).group(1)
        temp_f = f
        new_f = fix_filename(f)
        if f == new_f:
            temp = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            images.append(cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY))
            labels.append(song)
        else:
            os.rename(temp_f, new_f)
            temp = cv2.imread(new_f, cv2.IMREAD_UNCHANGED)
            images.append(cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY))
            labels.append(song)
            os.rename(new_f, temp_f)
            

    images = np.array(images)
    return images, labels

def fix_filename(filename):
    """Replaces non-alphanumeric characters in a filename with underscores."""
    return re.sub(r'[^a-zA-Z0-9]', '_', filename)

def setfile_at_directory(directory):
    """Renames files in a directory, replacing non-alphanumeric characters with underscores."""
    for filename in os.listdir(directory):
        new_filename = fix_filename(filename)
        if new_filename != filename:
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f"Renamed '{filename}' to '{new_filename}'")

def train():
    epochs = 16
    num_classes = 8
    batch_size = 128
    learning_rate = 0.001

    trainDataset = TrainDataset()
    train_loader = DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True, num_workers=0)
    testDataset = TestDataset()
    test_loader = DataLoader(dataset=testDataset, batch_size=batch_size, shuffle=True)

    train_model = model.to(device)
    summary(train_model, input_size=(1, 128, 128))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=learning_rate, weight_decay=0.01)

    total_samples = len(trainDataset)
    n_iterations = math.ceil(total_samples/batch_size)
    best_accuracy = 0
    for epoch in range(epochs):
        train_model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = (images/255.).reshape(-1, 1, 128, 128).to(device)
            labels = F.one_hot(labels.type(torch.LongTensor), num_classes).reshape(-1, num_classes).to(device)

            preds = train_model(images)

            loss = loss_fn(preds, torch.max(labels, 1)[1])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(preds, 1)
            n = argmax.size(0)
            accuracy = (torch.max(labels, 1)[1] == argmax).sum().item()/n

            if accuracy > best_accuracy:
                best_model = copy.deepcopy(model)
            
            if (i+1) % 100 == 0:
                print(f'epoch: {epoch}/{epochs}, step: {i+1}/{n_iterations}, loss: {loss.item():.4f}, accuracy: {accuracy:.4f}')
        
        train_model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = (images/255.).reshape(-1, 1, 128, 128).to(device)
                labels = F.one_hot(labels.type(torch.LongTensor), num_classes).reshape(-1, num_classes).to(device)
                preds = train_model(images)
                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == torch.max(labels,1)[1]).sum().item()
            
            print(f'validation accuracy: {(100 * correct/total):.4f} %')
    
    torch.save(best_model.state_dict(), 'model.pt')

def recommend(song, images, labels, rec_model):
    vector_size = 40

    recommend_song = song
    pred_anchor = torch.zeros((1, vector_size)).to(device)
    count = 0
    preds_song = []
    preds_label = []
    counts = []
    dist_array = []

    with torch.no_grad():
        for i in range(0, len(labels)):
            # Sum latent vector of anchor song
            if (labels[i] == recommend_song):
                test_image = images[i]
                test_image = test_image[None, :, :, :]
                trans_image = torch.from_numpy(test_image.astype(np.float32)).to(device)
                pred = rec_model(trans_image)
                pred_anchor = pred_anchor + pred
                count = count + 1
            # Add new prediction to list
            elif(labels[i] not in preds_label):
                preds_label.append(labels[i])
                test_image = images[i]
                test_image = np.expand_dims(test_image, axis=0)
                trans_image = torch.from_numpy(test_image.astype(np.float32)).to(device)
                pred = rec_model(trans_image)
                preds_song.append(pred)
                counts.append(1)
            # Sum latent feature vector for each predicted song
            elif(labels[i] in preds_label):
                index = preds_label.index(labels[i])
                test_image = images[i]
                test_image = np.expand_dims(test_image, axis=0)
                trans_image = torch.from_numpy(test_image.astype(np.float32)).to(device)
                pred = rec_model(trans_image)
                preds_song[index] = preds_song[index] + pred
                counts[index] = counts[index] + 1
        pred_anchor = pred_anchor / count # average latent feature vector of the anchor song
        for i in range(len(preds_song)):
            preds_song[i] = preds_song[i] / counts[i] # average latent feature vector for predicted song
            cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
            dist_array.append(cosine(pred_anchor, preds_song[i]))
        dist_array = torch.tensor(dist_array)
        recommendations = 2
        list_song = []
        chosen_song = {"id": 1, "name": recommend_song, "file": "./song_db/" + recommend_song + ".mp3", "type": "Original Song"}
        list_song.append(chosen_song)

        while recommendations < (int(args.recommendations)+2):
            index = torch.argmax(dist_array)
            value = dist_array[index]
            print("Song Name: " + "'" + preds_label[index] + "'" + " with value = %f" % (value))
            value = '{:.4f}'.format(value.item())
            list_song.append({"id": recommendations, "name": preds_label[index], "file": "./song_db/" + preds_label[index] + ".mp3", "type": "Recommended Song", "metric": "Similarity", "value": value})
            dist_array[index] = float('-inf')
            recommendations += 1
        return list_song
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Song recommender and trainer")
    argparser.add_argument("-m", "--mode", required=True, help='set mode to Train or Recommend')
    argparser.add_argument("-s", "--song", required=False, help='Recommend similar songs to given song')
    argparser.add_argument("-r", "--recommendations", required=False, default=3, help="set the number of recommendations to be returned")
    args = argparser.parse_args()
    if args.mode == "Recommend":
        if args.song is None:
            print("Please provide a song using the -s argument")
            exit
        song = ' '
        SONG_OPTIONS, images, labels, rec_model = load_data()
        if args.song not in np.unique(labels):
            print("Please provide a valid song name")
            print(SONG_OPTIONS)
            exit
        else:
            recommend_dict = recommend(args.song, images, labels, rec_model)
            print(recommend_dict)
    elif args.mode == "Train":
        train()