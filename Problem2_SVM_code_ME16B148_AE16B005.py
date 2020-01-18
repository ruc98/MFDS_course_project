### Problem02  ###

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

## SVM_func() passes the required arguments to the SVC function from sklearn according to the kernel
def SVM_func(X_train, Y_train, kernel, C, kernel_param):
    if kernel == 'linear':
        SVM_algo =  SVC(C=C, kernel=kernel)
    if kernel == 'poly':
        SVM_algo =  SVC(C=C, kernel=kernel, degree = kernel_param)
    else:
        SVM_algo =  SVC(C=C, kernel=kernel, gamma = kernel_param)

    classifier = SVM_algo.fit(X_train,Y_train)
    # returns the classifier object
    return classifier


## best_hyperparam() finds the best hyperparameters like regularization, polynomial degree and lmbda value for different kernels
def best_hyperparam(X_train, Y_train, kernel):
    split = int(0.8*X_train.shape[0])   #splitting the data into train and validation
    X_train1 = X_train[:split]
    X_val = X_train[split:]
    Y_train1 = Y_train[:split]
    Y_val = Y_train[split:]
    best_zero_one_loss = 1000
    best_kernel_param = 1
    best_reg_param = 0
    
    reg_params = [1e-5,1e-2,1]      #range of possible regularization parameters for all three kernels
    if kernel == 'linear':
        kernel_param = 1
        for C in reg_params:
            classifier = SVM_func(X_train1, Y_train1, kernel, C, kernel_param)
            Y_val_pred1 = classifier.decision_function(X_val)
            Y_val_pred = np.where(Y_val_pred1>0,1,0)
            zero_one_loss = np.where(Y_val_pred != Y_val,1,0)
            mean_zero_one_loss = np.mean(zero_one_loss)
            if mean_zero_one_loss < best_zero_one_loss:
                best_zero_one_loss = mean_zero_one_loss
                best_kernel_param = kernel_param
                best_reg_param = C
#             print('C ',C,'loss = ',mean_zero_one_loss)
            
    degree_params = [4,5,7,9,12]    # degree parameters range        
    if kernel =='poly':  
        for kernel_param in degree_params:
            for C in reg_params:
                classifier = SVM_func(X_train1, Y_train1, kernel, C, kernel_param)
                Y_val_pred1 = classifier.decision_function(X_val)
                Y_val_pred = np.where(Y_val_pred1>0,1,0)
                zero_one_loss = np.where(Y_val_pred != Y_val,1,0)
                mean_zero_one_loss = np.mean(zero_one_loss)
                if mean_zero_one_loss < best_zero_one_loss:
                    best_zero_one_loss = mean_zero_one_loss
                    best_kernel_param = kernel_param
                    best_reg_param = C
#                 print('C ',C,'degree ', kernel_param, 'loss = ',mean_zero_one_loss)


    rbf_lambda_params = [1e-3,1e-1,1,1e2] # rbf lambda range
    if kernel =='rbf':
        for kernel_param in rbf_lambda_params:
            for C in reg_params:
                classifier = SVM_func(X_train1, Y_train1, kernel, C, kernel_param)
                Y_val_pred1 = classifier.decision_function(X_val)
                Y_val_pred = np.where(Y_val_pred1>0,1,0)
                zero_one_loss = np.where(Y_val_pred != Y_val,1,0)
                mean_zero_one_loss = np.mean(zero_one_loss)
                if mean_zero_one_loss < best_zero_one_loss:
                    best_zero_one_loss = mean_zero_one_loss
                    best_kernel_param = kernel_param
                    best_reg_param = C
#                 print('C ',C,'lambda ', kernel_param, 'loss = ',mean_zero_one_loss)

    # returns the best hyperparameters parameters                   
    return best_kernel_param, best_reg_param


train_full = pd.read_csv("q2_data_matrix.csv")
labels = pd.read_csv("q2_labels.csv")
X_full_ = np.array(train_full)
Y_full_ = np.ravel(np.array(labels))
np.random.seed(10)
ind_list = [i for i in range(X_full_.shape[0])]
np.random.shuffle(ind_list)     # data shuffling
X_full = X_full_[ind_list,:]
Y_full = Y_full_[ind_list,]


split = int(0.8*X_full.shape[0])    # train-test split
X_train = X_full[:split]
X_test = X_full[split:]
Y_train = Y_full[:split]
Y_test = Y_full[split:]

kernels = ['poly','rbf','linear']
for kernel in kernels:                      # analysing different kernels
    print('Analysis of '+kernel+' kernel')
    ## finding best params ###
    print('Hyperparameter tuning...')
    best_kernel_param, best_reg_param = best_hyperparam(X_full, Y_full, kernel)     # finds best hyperparameters

    if kernel != 'linear':
        print("Best kernel_param = ",best_kernel_param)
    print("Best reg_param = ",best_reg_param)
    print(' ')

    ### train and test on best params ###
    classifier = SVM_func(X_train, Y_train, kernel, C = best_reg_param, kernel_param = best_kernel_param) # training with best hyperparameters parameters


    Y_train_pred = np.where(classifier.decision_function(X_train)>0,1,0)
    Y_test_pred = np.where(classifier.decision_function(X_test)>0,1,0)
    train_zero_one_loss = np.mean(np.where(Y_train_pred != Y_train,1,0))    # loss and accuracies
    test_zero_one_loss = np.mean(np.where(Y_test_pred != Y_test,1,0))
    train_acc = 1 - train_zero_one_loss
    test_acc = 1 - test_zero_one_loss
#     print('For '+kernel+' kernel, ')
    print('For Best parameters of {} kernel'.format(kernel))
    print("train loss = {0:.4f}, test loss = {1:.4f} ".format(train_zero_one_loss, test_zero_one_loss))
    print("train accuracy = {0:.4f} , test accuracy = {1:.4f} ".format(train_acc,test_acc))
    print(' ')


# now for Linear Kernel, calculating Confusion Matrix and F1 Score
kernel = 'linear'
best_kernel_param = 1
best_reg_param = 1e-5

classifier = SVM_func(X_train, Y_train, kernel, C = best_reg_param, kernel_param = best_kernel_param)


Y_train_pred = np.where(classifier.decision_function(X_train)>0,1,0)
Y_test_pred = np.where(classifier.decision_function(X_test)>0,1,0)


cm_train = confusion_matrix(Y_train, Y_train_pred)      # Confusion matrix and F1 score from sklearn.metrics
cm_test = confusion_matrix(Y_test, Y_test_pred)
f1_train = f1_score(Y_train, Y_train_pred)
f1_test = f1_score(Y_test, Y_test_pred)

print('For Linear Kernel..')
print('Confusion Matrix for Train Data')
print(cm_train)
print('Confusion Matrix for Test Data')
print(cm_test)
print('F1 Score for Train Data')
print(f1_train)
print('F1 Score for Test Data')
print(f1_test)

print('Done!!')
