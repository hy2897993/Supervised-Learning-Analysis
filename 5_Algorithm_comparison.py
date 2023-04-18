#############################################################################
#%%
import random
import time
import librosa
import soundfile
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import os

cwd = os.getcwd()
cwd = cwd.replace('\\','\\\\')
print(cwd)


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
    for file in glob.glob(cwd+"\\speech-emotion-recognition-ravdess-data\\Actor_*\\*.wav"):
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
X_ser,y_ser=load_data()

X_0, y_0 = fetch_covtype(return_X_y=True)
X_fct, X_1, y_fct, y_1 = train_test_split(X_0, y_0, test_size=0.995, random_state=42)



#############################################################################
#%%
def learning_algo_comparison(
    X_ser,y_ser,
    X_fct, y_fct,
    ylim=None
):


    _, axes = plt.subplots(2, 1, figsize=(15, 20))

    axes[0].set_title("Learning Algorithm Accuracy")
    if ylim is not None:
        axes[0].set_ylim(*ylim)
   
    algos = ["Decision Tree", "Neural Network", "K-Nearest Neighbor", "Boosting", "SVMs"]
    axes[0].set_ylabel("Accuracy Score")

    
    test_scores_ser = [0,0,0,0,0]
    test_scores_fct =[0,0,0,0,0]
    time_spend_ser = [0,0,0,0,0]
    time_spend_fct = [0,0,0,0,0]
    
    # Decision Tree
    #ser
    t_00 = time.time()
    clf_ser = DecisionTreeClassifier(ccp_alpha = 0.013851998)
    DT_ser_score = cross_val_score(clf_ser, X_ser,y_ser, cv=5)
    
    t_01 = time.time()-t_00
    test_scores_ser[0] = sum(DT_ser_score) / len(DT_ser_score)
    time_spend_ser[0] = t_01
    
    #fct
    t_02 = time.time()
    clf_fct = DecisionTreeClassifier(ccp_alpha = 0.013851998)
    DT_fct_score = cross_val_score(clf_fct, X_fct, y_fct, cv=5)
    
    t_03 = time.time()-t_02
    test_scores_fct[0] = sum(DT_fct_score) / len(DT_fct_score)
    time_spend_fct[0] = t_03
    
    # Neural Network
    # ser
    t_04 = time.time()
    clf_ser = MLPClassifier(hidden_layer_sizes=(200,100,50,25), random_state=1)
    nn_ser_score = cross_val_score(clf_ser, X_ser,y_ser, cv=5)
    
    t_05 = time.time()-t_04
    test_scores_ser[1] = sum(nn_ser_score) / len(nn_ser_score)
    time_spend_ser[1] = t_05
    
    #fct
    t_06 = time.time()
    clf_fct = MLPClassifier(hidden_layer_sizes=(200,100,50,25), random_state=1)
    nn_fct_score = cross_val_score(clf_fct, X_fct, y_fct, cv=5)
    
    t_07 = time.time()-t_06
    test_scores_fct[1] = sum(nn_fct_score) / len(nn_fct_score)
    time_spend_fct[1] = t_07
    
    # KNN
    # ser
    t_08 = time.time()
    clf_ser = KNeighborsClassifier(weights = 'distance',algorithm = 'auto', n_neighbors=2)
    knn_ser_score = cross_val_score(clf_ser, X_ser,y_ser, cv=5)
    
    t_09 = time.time()-t_08
    test_scores_ser[2] = sum(knn_ser_score) / len(knn_ser_score)
    time_spend_ser[2] = t_09
    
    #fct
    t_10 = time.time()
    clf_fct = KNeighborsClassifier(weights = 'distance',algorithm = 'auto', n_neighbors=3)
    knn_fct_score = cross_val_score(clf_fct, X_fct, y_fct, cv=5)
    
    t_11 = time.time()-t_10
    test_scores_fct[2] = sum(knn_fct_score) / len(knn_fct_score)
    time_spend_fct[2] = t_11
    
    # Boosting
    # ser
    t_12 = time.time()
    clf_ser = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=0)
    boosting_ser_score = cross_val_score(clf_ser, X_ser,y_ser, cv=5)
    
    t_13 = time.time()-t_12
    test_scores_ser[3] = sum(boosting_ser_score) / len(boosting_ser_score)
    time_spend_ser[3] = t_13
    
    #fct
    t_14 = time.time()
    clf_fct = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=0)
    boosting_fct_score = cross_val_score(clf_fct, X_fct, y_fct, cv=5)
    
    t_15 = time.time()-t_14
    test_scores_fct[3] = sum(boosting_fct_score) / len(boosting_fct_score)
    time_spend_fct[3] = t_15
    
    
    # SVMs
    # ser
    t_16 = time.time()
    clf_ser = svm.SVC(C = 5000, kernel="rbf")
    SVMs_ser_score = cross_val_score(clf_ser, X_ser,y_ser, cv=5)
    
    t_17 = time.time()-t_16
    test_scores_ser[4] = sum(SVMs_ser_score) / len(SVMs_ser_score)
    time_spend_ser[4] = t_17
    
    #fct
    t_18 = time.time()
    clf_fct = svm.SVC(C = 5000, kernel="rbf")
    SVMs_fct_score = cross_val_score(clf_fct, X_fct, y_fct, cv=5)
    
    t_19 = time.time()-t_18
    test_scores_fct[4] = sum(SVMs_fct_score) / len(SVMs_fct_score)
    time_spend_fct[4] = t_19
    
    
    print(test_scores_ser)
    print(time_spend_ser)
    print(test_scores_fct)
    print(time_spend_fct)
    
#     data = [[30, 25, 50, 20, 15],[40, 23, 51, 17, 30]]
#     X = np.arange(5)
#     axes[0].bar(X + 0.00, data[0], color = 'b', width = 0.25)
#     axes[0].bar(X + 0.25, data[1], color = 'y', width = 0.25)
    X = np.arange(5)
    
    axes[0].grid()

    axes[0].bar(X + 0.00, test_scores_ser, color = 'r', width = 0.25)
    axes[0].bar(X + 0.25, test_scores_fct, color = 'g', width = 0.25)
    
    
    
    #plot time curve
    axes[1].set_title("Learning Algorithm Time Cost")
   
    axes[1].set_ylabel("Time spent")
    
    axes[1].grid()

    axes[1].bar(X + 0.00, time_spend_ser, color = 'b', width = 0.25)
    axes[1].bar(X + 0.25, time_spend_fct, color = 'y', width = 0.25)

    return plt


#############################################################################
#%%
learning_algo_comparison(
    X_ser,y_ser,
    X_fct, y_fct
)

# %%
