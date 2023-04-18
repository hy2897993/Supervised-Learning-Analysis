#############################################################################
#%%
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier



#############################################################################
#%%

X_0, y_0 = fetch_covtype(return_X_y=True)
X, X_1, y, y_1 = train_test_split(X_0, y_0, test_size=0.995, random_state=42)

#############################################################################
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)




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
        
        test_scores.append(clf.score(X_test, y_test))
        train_scores.append(clf.score(X, y))
        
    for i in iterations:
        
        clf = GradientBoostingClassifier(n_estimators=i, learning_rate=0.1, max_depth=1, random_state=0,ccp_alpha = ccp_alpha).fit(X, y)
        
        test_scores_pruned.append(clf.score(X_test, y_test))
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
    X_train,
    y_train,
    ccp_alpha = 0.0001,
    iterations=range(1,150, 5),
    ylim=(0.4, 0.8))


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
        
        test_scores_pruned.append(clf.score(X_test, y_test))
        train_scores_pruned.append(clf.score(X, y))
        
    for d in depth:
        
        clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=d, random_state=0).fit(X, y)
        
        test_scores.append(clf.score(X_test, y_test))
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
    X_train,
    y_train,
    ccp_alpha = 0.0001,
    depth=range(1,10),
    ylim=(0, 1.01))