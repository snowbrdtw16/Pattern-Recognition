### In order to run this code all you need to do is go to the bottom of the code and call the function bayes_classifier. It takes in the following arguments:
# "Train" = Path to the training data
# "eval_data" = Path to the dataset you want to evaluate.
# "question" = int 1-5, for the scenario you want to run (If you want to change the features for 5 you will need to change some of the code internally to pick the correct features.)
# "d" = Number of dimensions (should always be = 8)

import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def bayes_classifier(train, eval_data, question, d):
# The variable X will be known as the feature matrix. Each row will be one feature vector.
    X = np.genfromtxt(train, skip_header=1, usecols=[1,2,3,4,5,6,7,8])
    X_lab = np.genfromtxt(train, dtype=str, skip_header=1, usecols=0)
    X_labu = np.unique(X_lab)
# 'classes' variable is the number of possible classes.
    classes = 10
# 'd' is the number of dimensions.
    RMS = np.sqrt(np.mean(X**2, axis=0))
    eval_dat = np.genfromtxt(eval_data, skip_header=1, usecols=[1,2,3,4,5,6,7,8])
    eval_dat = eval_dat/RMS
    eval_lab = np.genfromtxt(eval_data, dtype=str, skip_header=1, usecols=0)
    X = X/RMS


    class_arr = []
    for i in range(classes):
        class_arr.append(X[i*10: (i+1)*10,:])
    
    class_arr = np.array(class_arr)

    # Class arr contains the 10 arrays with the shape (row, cols), (10, 8). There are 10 samples (rows) per class and there are 8 features (Columns) per class.

    # mu is the average value of each feature for each class. (10x8 array, 10 classes x 8 features)
    mu = np.mean(class_arr, axis=1)

    
    
    ## Here is where I use the identity matrix as the covariance matrix.
    
    
    
    if question == 1:
        Cov_mat = np.identity(d)
        results = []
        for i in np.arange(eval_dat.shape[0]):
            g = []
            for j in np.arange(classes):
                xi = eval_dat[i, np.newaxis]
                mu_j = mu[j, np.newaxis]
                delta = xi - mu_j
                deter = np.linalg.det(Cov_mat)
                Cov_inv = np.linalg.inv(Cov_mat)
                discrim = (-1/2)*np.matmul(delta, np.matmul(Cov_inv, delta.T)) - (1/2)*np.log(deter)
                g.append(discrim)
            pick = X_labu[np.argmax(g)]
            results.append(pick)
        results = np.array(results, ndmin=2).transpose()
        
    
    
    
    
    ## Here is the portion of code that uses the average covariance matrix.
    
    
    if question == 2:
        Cov_mat = []
        for i in range(classes):
            Cov_mat.append(np.cov(class_arr[i,:,:], rowvar=False))
        Cov_mat = np.array(Cov_mat).mean(axis=0)
        results = []
        for i in np.arange(eval_dat.shape[0]):
            g = []
            for j in np.arange(classes):
                xi = eval_dat[i, np.newaxis]
                mu_j = mu[j, np.newaxis]
                delta = xi - mu_j
                deter = np.linalg.det(Cov_mat)
                Cov_inv = np.linalg.inv(Cov_mat)
                discrim = (-1/2)*np.matmul(delta, np.matmul(Cov_inv, delta.T)) - (1/2)*np.log(deter)
                g.append(discrim)
            pick = X_labu[np.argmax(g)]
            results.append(pick)
        results = np.array(results, ndmin=2).transpose()
        

    
   
    
    # This portion of code is for when we use the individual covariance matrices for each class.
    
    
    
    if question == 3:
        Cov_mat = []
        for i in range(classes):
            Cov_mat.append(np.cov(class_arr[i,:,:], rowvar=False))
        Cov_mat = np.array(Cov_mat)
        
        results = []
        for i in np.arange(eval_dat.shape[0]):
            g = []
            for j in np.arange(classes):
                xi = eval_dat[i,:, np.newaxis]
                mu_j = mu[j,:,np.newaxis]
                delta = xi - mu_j
                deter = np.linalg.det(Cov_mat[j,:,:])
                Cov_inv = np.linalg.inv(Cov_mat[j,:,:])
                discrim = (-1/2)*np.matmul(delta.T, np.matmul(Cov_inv, delta)) - (1/2)*np.log(deter)
                g.append(discrim)
            pick = X_labu[np.argmax(g)]
            results.append(pick)
        results = np.array(results, ndmin=2).transpose()
    
    
    
    
   ## This portion of the code is for when we only use the first 4 features in the training set. 
    
    
    if question == 4:
        Cov_mat = []
        for i in range(classes):
            Cov_mat.append(np.cov(class_arr[i,:,0:4], rowvar=False))
        Cov_mat = np.array(Cov_mat)
        results = []
        print(mu[:,0:4, np.newaxis])
        for i in np.arange(eval_dat.shape[0]):
            g = []
            for j in np.arange(classes):
                xi = eval_dat[i,0:4, np.newaxis]
                mu_j = mu[j,0:4,np.newaxis]
                delta = xi - mu_j
                deter = np.linalg.det(Cov_mat[j,:,0:4])
                Cov_inv = np.linalg.inv(Cov_mat[j,:,0:4])
                discrim = (-1/2)*np.matmul(delta.T, np.matmul(Cov_inv, delta)) - (1/2)*np.log(deter)
                g.append(discrim)
            
            pick = X_labu[np.argmax(g)]
            results.append(pick)
        results = np.array(results, ndmin=2).transpose()
        
    
    
## This next code is for the graduate student portion. I decided to use the features Mu11 and Mu30.     
    
    
    if question == 5:
        Cov_mat = []
        for i in range(classes):
            Cov_mat.append(np.cov(class_arr[i,:,5::2], rowvar=False))
        Cov_mat = np.array(Cov_mat)
        results = []
        for i in np.arange(eval_dat.shape[0]):
            g = []
            for j in np.arange(classes):
                xi = eval_dat[i,5::2, np.newaxis]
                mu_j = mu[j,5::2,np.newaxis]
                delta = xi - mu_j
                deter = np.linalg.det(Cov_mat[j,:])
                Cov_inv = np.linalg.inv(Cov_mat[j,:])
                discrim = (-1/2)*np.matmul(delta.T, np.matmul(Cov_inv, delta)) - (1/2)*np.log(deter)
                g.append(discrim)
            
            pick = X_labu[np.argmax(g)]
            results.append(pick)
        results = np.array(results, ndmin=2).transpose()


   
    
    confus = confusion_matrix(eval_lab, results, labels=X_labu)

    error_II =  np.array(np.sum(confus, axis=0) - np.diagonal(confus), ndmin=2)
    error_I = np.array(np.sum(confus, axis=1) - np.diagonal(confus), ndmin=2)

    error_I = np.array(np.append(error_I, [0]), ndmin=2)


    confus = np.append(confus, error_II, axis=0)
    confus = np.append(confus, error_I.T, axis=1)
    accuracy = np.sum(np.diagonal(confus)) / eval_dat.shape[0]


## Plot results
    
    
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    ax = sns.heatmap(confus, vmax=eval_dat.shape[0]/classes, xticklabels=np.append(X_labu, ('Error I')),            yticklabels=np.append(X_labu, ('Error II')), cbar=False, linewidths=1, linecolor='black', cmap='Blues', annot=True)
    ax.set_xlabel('Decision')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Input')
    ax.text(0, -0.1, 'Accuracy = {:.2f}%'.format(accuracy*100),
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes)
    ax.text(0.8, -0.1, 'Error = {:.2f}%'.format((1-accuracy)*100),
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes)
    return ax


# In order to run the program, you need to use the 'bayes_classifier' function. It takes in the trainingdat.txt, eval set, the situation you want to run (int from 1-5), and the dimensions (8 in this case)
bayes_classifier('traindat.txt', 'eval2dat.txt', 5, 8)




#print(1/((2*np.pi)**(d/2))*deter**(1/2))

# xi = np.array(eval_dat[0,:], ndmin=2)
# mu1 = np.array(mu[0,:], ndmin=2)
#
# print((xi - mu1).shape)
# print(np.linalg.inv(Cov_mat[0,:,:]).shape)
# term1 = np.matmul((xi - mu1),np.linalg.inv(Cov_mat[0,:,:]))
# print(np.matmul(term1, np.transpose(xi - mu1)))
