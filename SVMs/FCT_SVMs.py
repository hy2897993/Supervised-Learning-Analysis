#############################################################################
#%%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn import svm


#############################################################################
#%%
X_0, y_0 = fetch_covtype(return_X_y=True)
X, X_1, y, y_1 = train_test_split(X_0, y_0, test_size=0.999, random_state=42)



#############################################################################
#%%
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


#############################################################################
#%%
def training_size_learning_curve(
    title,
    X,
    y,
    kernel_func,
    train_sizes=np.linspace(0.1, 1.0, 5),
    ylim=None
):
    


    plt.suptitle(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training size")
    plt.ylabel("Score")
    
    
    train_scores = []
    test_scores = []
    
    for size in train_sizes:
        
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size = size, random_state=42)
        
        #Create a svm Classifier
        clf = svm.SVC(C = 5000, kernel=kernel_func) # Linear Kernel
# , max_iter = 1000
        #Train the model using the training sets
        clf.fit(X_tr, y_tr)
        
        y_pred_train = clf.predict(X_tr)
        y_pred_test = clf.predict(x_test)
        
        train_scores.append(accuracy_score(y_true=y_tr, y_pred=y_pred_train))
        test_scores.append(accuracy_score(y_true=y_test, y_pred=y_pred_test))
        
    
    # Plot learning curve
    plt.grid()

    plt.plot(
        train_sizes, test_scores, "o-", color="g",markersize=5, label="Test score"
    )
    
    plt.plot(
        train_sizes, train_scores, "o-", color="r",markersize=5, label="Train score"
    )
    plt.legend(loc="best")
    
    return plt

#############################################################################
#%%
title = "Learning Curves of training size"


#takes forever, have to set max iteration 3000
training_size_learning_curve(
    title,
    x_train,
    y_train,
    kernel_func='sigmoid',
    train_sizes=np.linspace(0.01, 0.99, 20),
    ylim=(0,1.01))


#############################################################################
#%%
def Kernel_function_curve(
    title,
    X,
    y,
    kernel_functions = ["rbf", "poly", "sigmoid"],
    Regularization_parameter = np.linspace(0.1, 10000, 500),
    ylim=None
):

    plt.suptitle(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Penalty parameter of the error term")
    plt.ylabel("Score")
    

    test_scores = [[],[],[]]
    
    for kernel_index in range(len(kernel_functions)):
        
        for i in Regularization_parameter:
            
            clf = svm.SVC(C = i, kernel=kernel_functions[kernel_index])
            clf.fit(X, y)
            
            y_pred = clf.predict(x_test)
            
            test_scores[kernel_index].append(accuracy_score(y_true=y_test, y_pred=y_pred))
            
    
    # Plot learning curve
    plt.grid()
    colors = ["r", "g", "b"]
    for kernel_index in range(len(kernel_functions)):
        plt.plot(
            Regularization_parameter, test_scores[kernel_index], "o-", color=colors[kernel_index], markersize=0.1, label= kernel_functions[kernel_index] + "score"
        )
    
    plt.legend(loc="best")
    
    return plt
            
        
        



#############################################################################
#%%
title = "Kernel functions of regularization parameter"


Kernel_function_curve(
    title,
    x_train,
    y_train,
    kernel_functions = ["rbf", "poly", "sigmoid"],
    Regularization_parameter = np.linspace(0.1, 1500, 50),
    ylim=(0,1.01))


