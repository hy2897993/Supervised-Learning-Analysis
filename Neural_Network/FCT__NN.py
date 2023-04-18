#############################################################################
#%%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split






#############################################################################
#%%
X_0, y_0 = fetch_covtype(return_X_y=True)
X, X_1, y, y_1 = train_test_split(X_0, y_0, test_size=0.995, random_state=42)

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
    learning_rate='adaptive', max_iter=3000
)


plot_learning_curve(
    estimator, title, x_train, y_train, 
    ylim=(-0.01,1.01), 
    train_sizes=np.linspace(0.01, 0.999,10), job = 10)


# %%
