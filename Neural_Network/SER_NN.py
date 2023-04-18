#%%
import random
import librosa
import soundfile
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
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
def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    ylim=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
    job = 10
):


    plt.suptitle(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores = []
    test_scores = []
    for size in train_sizes:
        train_score_i = 0
        test_score_i = 0
        for i in range(job):
            clf = estimator
            x_tr,x_te,y_tr,y_te=train_test_split(np.array(X), y, train_size=size, random_state=i^2+i)
            clf.fit(x_tr, y_tr)
        
            y_tr_pred=clf.predict(x_tr)
            y_test_pred=clf.predict(x_test)
        
            accuracy_tr=accuracy_score(y_true=y_tr, y_pred=y_tr_pred)
            accuracy_test=accuracy_score(y_true=y_test, y_pred=y_test_pred)
            
            train_score_i = train_score_i + accuracy_tr
            test_score_i = test_score_i + accuracy_test

        train_score_i = train_score_i/job
        test_score_i = test_score_i/job
        
        train_scores.append(accuracy_tr)
        test_scores.append(accuracy_test)





    # Plot learning curve
    plt.grid()

    plt.plot(
        train_sizes, train_scores, "o-", color="r", label="Training score"
    )
    plt.plot(
        train_sizes, test_scores, "o-", color="g", label="Test score"
    )
    plt.legend(loc="best")


    return plt



#############################################################################
#%%
title = "Learning Curves (Neural Network)"

estimator = MLPClassifier(
    alpha=0.01, 
    batch_size='auto', 
    epsilon=1e-08, 
    hidden_layer_sizes=(200,100,50,25), 
    learning_rate='adaptive'
)


plot_learning_curve(
    estimator, title, x_train, y_train, 
    ylim=(-0.01,1.01), 
    train_sizes=np.linspace(0.01, 0.999,10), job = 10)


# %%
