#%%

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
#######################################################################################################################
#%%

X_0, y_0 = fetch_covtype(return_X_y=True)
X, X_1, y, y_1 = train_test_split(X_0, y_0, test_size=0.995, random_state=42)
###########################################################################################################
#%%

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
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    

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

###############################################################################################   
#%%

title = "Learning Curves (Decision Tree)"


cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

estimator = DecisionTreeClassifier(random_state=0, ccp_alpha = 0.013851998)


plot_learning_curve(
    estimator, title, X, y, 
    ylim=(0.55, 0.85), 
    cv=cv, n_jobs=4,  train_sizes=np.linspace(0.1, 1.0,10))


#######################################################################################
#%%
def plot_iteration_curve(
    title,
    X,
    y,
    ylim=None,
    depth=range(2,51),
    test_size=0.4
):
    

    plt.suptitle(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Max Depth")
    plt.ylabel("Score")

    

    train_scores = [0]
    test_scores = [0]
   
    for i in depth:
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=5) 
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
###################################################################################################
#%%
plot_iteration_curve("Model Complexity", X, y, 
    ylim=(-0.01, 1.02), 
    depth=range(1,24),    test_size=0.3)
# %%
