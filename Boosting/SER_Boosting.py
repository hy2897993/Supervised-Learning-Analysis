#############################################################################
#%%
import random
import librosa
import soundfile
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

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
def boosting_iteration_curve(
    title,
    X,
    y,
    ccp_alpha = 0.001,
    iterations=range(1,101),
    ylim=None
):


    plt.suptitle(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Accuracy Score")

    
    test_scores = []
    train_scores = []
    test_scores_pruned = []
    train_scores_pruned = []
    

    for i in iterations:
        
        clf = GradientBoostingClassifier(n_estimators=i, learning_rate=0.1, max_depth=1, random_state=0).fit(X, y)
        
        test_scores.append(clf.score(x_test, y_test))
        train_scores.append(clf.score(X, y))
        
    for i in iterations:
        
        clf = GradientBoostingClassifier(n_estimators=i, learning_rate=0.1, max_depth=1, random_state=0,ccp_alpha = ccp_alpha).fit(X, y)
        
        test_scores_pruned.append(clf.score(x_test, y_test))
        train_scores_pruned.append(clf.score(X, y))
        


    print(test_scores)
    print(train_scores)

    # Plot accuracy/k-NN curve
    plt.grid()

    plt.plot(
        iterations, test_scores, "o-", color="g",markersize=0.1, label="Test score"
    )
    
    plt.plot(
        iterations, train_scores, "o-", color="r",markersize=0.1, label="Train score"
    )
    
    plt.plot(
        iterations, test_scores_pruned, "o-", color="y",markersize=0.1, label="Pruned Test score"
    )
    
    plt.plot(
        iterations, train_scores_pruned, "o-", color="b",markersize=0.1, label="Pruned Train score"
    )

    plt.legend(loc="best")



    return plt


#############################################################################
#%%
title = "Boosting Iteration Train/Test Score Curves"



boosting_iteration_curve(
    title,
    x_train,
    y_train,
    ccp_alpha = 0.001,
    iterations=range(1,150, 5))

#############################################################################
#%%
def boosting_depth_curve(
    title,
    X,
    y,
    ccp_alpha = 0.001,
    depth=range(1,10),
    ylim=None
):

    plt.suptitle(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Maximum Depth")
    plt.ylabel("Accuracy Score")

    
    test_scores = []
    train_scores = []
    test_scores_pruned = []
    train_scores_pruned = []

    for d in depth:
        
        clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=d, random_state=0,ccp_alpha = ccp_alpha).fit(X, y)
        
        test_scores_pruned.append(clf.score(x_test, y_test))
        train_scores_pruned.append(clf.score(X, y))
        
    for d in depth:
        
        clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=d, random_state=0).fit(X, y)
        
        test_scores.append(clf.score(x_test, y_test))
        train_scores.append(clf.score(X, y))
        


    print(test_scores)
    print(train_scores)

    # Plot accuracy/k-NN curve
    plt.grid()

    plt.plot(
        depth, test_scores, "o-", color="g",markersize=5, label="Test score"
    )
    
    plt.plot(
        depth, train_scores, "o-", color="r",markersize=5, label="Train score"
    )
    
    plt.plot(
        depth, test_scores_pruned, "o-", color="y",markersize=5, label="Pruned Test score"
    )
    
    plt.plot(
        depth, train_scores_pruned, "o-", color="b",markersize=5, label="Pruned Train score"
    )

    plt.legend(loc="best")

    


    return plt

#############################################################################
#%%
title = "Max Depth Train/Test Score Curves"



boosting_depth_curve(
    title,
    x_train,
    y_train,
    ccp_alpha = 0.0005,
    depth=range(1,10),
    ylim=(0, 1.01))