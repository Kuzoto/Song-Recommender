### Setup
-Download all requirements listed in requirements.txt</br>
***Make sure install the PyTorch with CUDA(Be aware your CUDA version may be newer than 12.8)***</br>
-Move into the Song-Recommender directory if you aren't already ***This is where you will run all commands***</br>
<img src="https://github.com/user-attachments/assets/a40fe7c3-b205-411a-b41c-e21556116559" width="20%" height="20%"></br>
-Download all fma small dataset containing all the training mp3s</br>
<img src="https://github.com/user-attachments/assets/e8015415-233e-40e4-b71f-7e576cf771a1" width="15%" height="15%"></br>
-Create the specs for both the training and recommendation songs</br>
<img src="https://github.com/user-attachments/assets/a6170399-c6bc-4508-b104-827f78199038" width="20%" height="20%"></br>
<img src="https://github.com/user-attachments/assets/f8ffd3f0-3d64-4807-9448-941803e05215" width="20%" height="20%"> ***(Test mode refers to recommendation songs)***</br>
-Slice the training and recommendation specs</br>
<img src="https://github.com/user-attachments/assets/7efd9dd0-459f-458e-b144-315b7f515830" width="20%" height="20%"></br>
<img src="https://github.com/user-attachments/assets/224cc062-775c-4a5c-96f7-010f277690b4" width="20%" height="20%"></br>
-Load the training data into training and/or test datasets</br>
<img src="https://github.com/user-attachments/assets/78a77687-c7cd-49cb-929b-b6b02ec715ca" width="20%" height="20%"> ***The training split size defaults to 1.0 meaning no test dataset***</br>
-Additionally, you can add more songs to the recommendation list using the following commands</br>
<img src="https://github.com/user-attachments/assets/8280075a-702a-43f6-8e16-383494261e6c" width="20%" height="20%"></br>
***The --url argument requires a valid spotify song url(PLEASE DO NOT PASS A PLAYLIST OR ALBUM URL, it will download the entire playlist/album but the rest of the behavior is undefined)***</br>
<img src="https://github.com/user-attachments/assets/cb69843b-af6b-4816-a850-e28a25e19269" width="20%" height="20%"></br>

### Usage</br>
***Run all commands in the root directory(i.e. Song-Recommender)***</br>
***Though it shouldn't cause any issues I recommend not running load_csv till you have ran specs and slices***</br>
***Double check your CUDA version and make sure that you have pytorch+cuda installed, this model trains and runs very slow on CPU***</br>
**You are provided a trained model save state(model.pt)**</br>
-If you wish to train the model, run the following command</br>
<img src="https://github.com/user-attachments/assets/212002df-9555-43ad-b99f-e2c92bc8c68c" width="20%" height="20%"></br>
-When you want to run the recommender, use the following command structure</br>
<img src="https://github.com/user-attachments/assets/4fdd1b78-1f1d-4d31-a506-5c4035c9e37d" width="20%" height="20%"></br>
***The song file name takes the form, artist(s) - song_name, and num_recommendations defaults to 3, mode should be Recommend***

### fma_dl.py</br>
This file downloads the mp3s used to train the model to produce recommendations. This script may take a while as you are downloading 8,000 thirty second mp3s. Though this downloads the mp3s for training it does not produce the spec slices inputed to the model and as such you must create them after downloading this data.

### spotify_specs.py</br>
This script creates the mel-spectrograms for the mp3s of whichever database you passed as an argument. This script has one functional argument -m, or --mode, and this argument takes either Train, for creating the specs of the training db, or Test, for creating the specs of the song db containing the recommendation songs.</br>

### spotify_slices.py</br>
This script takes the mel-spectrograms of the passed database and slices them into smaller 128x128 images. This script has one argument -m, or --mode, and this argument takes either Train or Test. An example of the input and output of this script is given below.</br>
<p align="center"><img src="https://github.com/user-attachments/assets/1340ad07-4f94-4bf0-8897-f8d1df1034ea" width="50%" height="50%"></p>

### load_training_data.py</br>
This script takes the training slices and their metadata, then creates npy datasets using them. This is how the model loads the training data when training. This script has one argument, which determines what percent of the training db will be split to the training dataset and the rest will be in the testing dataset.

### add_song.py</br>
This script downloads a song from a spotify url and then creates a mel-spectrogram using its mp3 and slices it. This script allows you to add individual songs to be used for recommendation. This script is not intended for downloading more than one song so please do not pass it an album or playlist url, as it only creates one mel-spectrogram.

### load_csv.py</br>
This script downloads ~8,000 songs evenly distributed among 114 genres and creates mel-specs and slices for each of them. This script pulls from a csv containing the metadata for ~89000 spotify tracks. When ran it samples 70 songs from each genre, so the songs added are somewhat random each time. This means that it can ran multiple times, if you wish to add more songs to the song_db(since it skips over duplicates without any errors).

### model.py</br>
This file contains the model declaration and design of our layers. The model is applied to all the slices of each song spectrogram. Average pooling was used to consider the entire feature vector(i.e. the entire song). Residual Blocking was used to prevent vanishing features. Batch normalization was used to prevent overfitting and stabilize training. ReLU was used as the activation function, preventing negative features. Conv2d was used to extract simple patterns and abstract features from the spectrogram. Dense layers were used to compress the feature vector into a latent feature vector which is then used to obtain a classification. Adaptive average pool to ensure that classification works regardless of length and Linear layers compress to latent feature vector and classification</br>
<p align="center"><img src="https://github.com/user-attachments/assets/a245903c-3521-4224-82a5-41d0f123e969" width="50%" height="50%"></p>

### recommender.py</br>
This file is used to both recommend songs and train the recommender, which function is ran is determined by the mode. For training, a learning rate of 0.001 was used, the optimization function AdamW was used with 0.01 weight decay, batch size was 128, the number of epochs was 16, and CrossEntropyLoss was used as the loss function. When ran in the recommend mode, the recommender removes the final Linear layer, so the model produces a latent feature vector instead of a classification. A latent feature vector is produced for each spectrogram slice and then the recommender goes through each songs’ slices summing their latent feature vectors. Once all the songs have been summed, their average latent feature vector is calculated. Then the cosine similarity of all songs to the anchor song is calculated and stored in an array. The recommendation list is comprised of the requested number of songs with the highest similarity scores. Below is an example function and it a visualization of the recommender.</br>
<img src="https://github.com/user-attachments/assets/d7b60f48-0c5e-46a8-b1d9-6546ec851dbc" width="50%" height="50%"></br>
<pre align="middle"><img src="https://github.com/user-attachments/assets/df583834-d979-4a10-a9f7-d93e5cf223fc" width="30%" height="30%">       <img src="https://github.com/user-attachments/assets/f2d130ed-1bed-4620-92ff-7956e0a8903d" width="30%" height="30%"></pre>

### summary.py</br>
This produces a summary of the model and the model used by the recommender.
