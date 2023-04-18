#############################################################################
#%%
import random
import librosa
import soundfile
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import time

import os

cwd = os.getcwd()
parent_d = os.path.abspath(os.path.join(cwd, os.pardir))
new_parent_d = parent_d.replace('\\','\\\\')
print(new_parent_d)

#############################################################################
#%%
# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])

        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))

        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))

        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))

    return result

    # Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}
emo_list = list(emotions.values())
# Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust']

# Load the data and extract features for each sound file
def load_data():
    x,y=[],[]
    for file in glob.glob(new_parent_d+"\\speech-emotion-recognition-ravdess-data\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        
        random_location = random.randint(0, len(x))
        x.insert(random_location, feature)
        y.insert(random_location, emo_list.index(emotion))
        
    #return train_test_split(np.array(x), y, test_size=test_size, random_state=6)
    return x,y



#############################################################################
#%%
X,y=load_data()


#############################################################################
#%%
x_train,x_test,y_train,y_test=train_test_split(np.array(X), y, test_size=0.25, random_state=9)


#############################################################################
#%%
def KNN_accuracy_curve(
    title,
    X,
    y,
    weights = "distance",
    k_neighbors=range(1,11),
    ylim=None
):
    _, axes = plt.subplots(2, 1, figsize=(15, 20))
    
    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Neighbor Counts")
    axes[0].set_ylabel("Accuracy Score")

    
    test_scores = []
    time_spend_train = []
    time_spend_test = []

    for k in k_neighbors:
        
        start_time = time.time()
        
        neigh = KNeighborsClassifier(weights = weights, n_neighbors=k)
        neigh.fit(X, y)
        
        
        time_spend_train.append(time.time() - start_time)
        fit_time = time.time()
        
        y_predict = neigh.predict(x_test)
        accuracy=accuracy_score(y_true=y_test, y_pred=y_predict)
        
        time_spend_test.append(time.time() - fit_time)
        
        test_scores.append(accuracy)

    print(test_scores)
    print(time_spend_train)
    print(time_spend_test)

    # Plot accuracy/k-NN curve
    axes[0].grid()

    axes[0].plot(
        k_neighbors, test_scores, "o-", color="r", label="Test score"
    )

    axes[0].legend(loc="best")
    
    #plot time_train/knn curve
        # Plot accuracy/k-NN curve
    axes[1].grid()

    axes[1].plot(
        k_neighbors, time_spend_train, "o-", color="r", label="Training Time"
    )
    axes[1].plot(
        k_neighbors, time_spend_test, "o-", color="g", label="Testing Time"
    )

    axes[1].legend(loc="best")
    
    axes[1].set_xlabel("Neighbor Counts")
    axes[1].set_ylabel("Time Spend")
    axes[1].set_title("K Nearest Neighbors Training/Test Time")
    


    return plt



#############################################################################
#%%
title = "K Nearest Neighbor Accuracy Curves"


KNN_accuracy_curve(
    title,
    x_train,
    y_train,
    weights = "distance",
    k_neighbors=range(1,11),
    ylim=(0.4,0.8))


#############################################################################
#%%
def KNN_weighted_neighbor(
    title,
    X,
    y,
    weights = ["distance","uniform"],
    k_neighbors=range(1,11),
    ylim=None
):

    plt.suptitle(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Neighbor Counts")
    plt.ylabel("Accuracy Score")

    
    
    scores = [[],[]]
    colors = ["b", "y"]

    # Plot accuracy/k-NN curve
    plt.grid()
    
    
        
    for i in range(len(weights)):
        for k in k_neighbors:
            neigh = KNeighborsClassifier(weights = weights[i],algorithm = 'auto', n_neighbors=k)
            neigh.fit(X, y)


            y_predict = neigh.predict(x_test)
            accuracy=accuracy_score(y_true=y_test, y_pred=y_predict)

            scores[i].append(accuracy)


        plt.plot(k_neighbors, scores[i], "o-", color=colors[i], label= weights[i]+ "weighted score")



    plt.legend(loc="best")
    
    


    return plt

#############################################################################
#%%
title = "K Nearest Neighbor Weight Curves"



KNN_weighted_neighbor(
    title,
    x_train,
    y_train,
    weights = ["distance","uniform"],
    k_neighbors=range(1,11),
    ylim=(0.4,0.8))
