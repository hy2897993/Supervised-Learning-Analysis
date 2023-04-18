
#%%

import random
import librosa
import soundfile
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os

cwd = os.getcwd()
parent_d = os.path.abspath(os.path.join(cwd, os.pardir))
new_parent_d = parent_d.replace('\\','\\\\')
print(new_parent_d)


fields = []

for x in range(40):
    fields += ["MFCC_" + str(x)]
    
for y in range(12):
    fields += ["Chroma_" + str(y)]
    
for z in range(128):
    fields += ["Mel_" + str(z)]

fields += ["emotion"]


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


###################################################################################################
# %%

X,y=load_data()

####################################################################################################
# %%
def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):


    plt.suptitle(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    print(train_scores_mean.shape,train_scores.shape )
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    plt.grid()
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    plt.legend(loc="best")

   

    return plt

####################################################################################################
# %%

title = "Learning Curves (Decision Tree)"


estimator = DecisionTreeClassifier(random_state=0, ccp_alpha = 0.013851998)


plot_learning_curve(
    estimator, title, X, y, 
    ylim=(0, 1.01), 
    cv=10, n_jobs=4,train_sizes=np.linspace(0.1, 1.0, 20))


####################################################################################################
# %%

def plot_iteration_curve(
    title,
    X,
    y,
    ylim=None,
    depth=range(2,51),
    test_size=0.4,
    random_state=0
):
    




    plt.suptitle(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Max Depth")
    plt.ylabel("Score")

    

    train_scores = [0]
    test_scores = [0]
   
    for i in depth:
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state) 
        clf = DecisionTreeClassifier(random_state=0, max_depth = i)
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        test_scores.append(accuracy_score(y_test, y_pred))
        train_scores.append(accuracy_score(y_train, y_pred_train))
        

    
    train_scores_std = np.std(train_scores)


    test_scores_std = np.std(test_scores)
    depth = list(depth)

    
    depth = [0] + depth

    

    # Plot learning curve
    plt.grid()
    plt.fill_between(
        depth,
        train_scores - train_scores_std,
        train_scores + train_scores_std,
        alpha=0.1,
        color="r",
    )

    plt.plot(
        depth, train_scores, "o-", color="r", label="Training set"
    )

    plt.fill_between(
        depth,
        test_scores - test_scores_std,
        test_scores + test_scores_std,
        alpha=0.1,
        color="g",
    )

    plt.plot(
        depth, test_scores, "o-", color="g", label="Test set"
    )

    plt.legend(loc="best")

    
    return plt


####################################################################################################
# %%

plot_iteration_curve("Model Complexity", X, y, 
    ylim=(-0.01, 1.02), 
    depth=range(1,18),    test_size=0.4, random_state =8)
# %%
