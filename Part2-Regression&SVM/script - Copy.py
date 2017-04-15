import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    print('In preprocess')
    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


"""
Main

"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()


"""
Script for Support Vector Machine
"""

print('\n\n------------SVM START-----------\n\n')
##################
# YOUR CODE HERE #
#print(train_label.shape)
#print(validation_label.shape)
#print(test_label.shape)
train_label = train_label.flatten()
validation_label = validation_label.flatten()
test_label = test_label.flatten()
#print(train_label.shape)
#print(validation_label.shape)
#print(test_label.shape)

print('\n***SVM with Linear Kernel***')
classifier_l = svm.SVC(kernel='linear')
classifier_l.fit(train_data,train_label)
accuracy = classifier_l.predict(train_data)
print('\nTraining set accuracy : ' + str(100 * np.mean((accuracy == train_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_l = svm.SVC(kernel='linear')
classifier_l.fit(validation_data,validation_label)
accuracy = classifier_l.predict(validation_data)
print('\nValidation set accuracy : ' + str(100 * np.mean((accuracy == validation_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_l = svm.SVC(kernel='linear')
classifier_l.fit(test_data,test_label)
accuracy = classifier_l.predict(test_data)
print('\nTest set accuracy : ' + str(100 * np.mean((accuracy == test_label).astype(float))) + "%")
#print('--------------------------------------')

print('\n***SVM with RBF Kernel and Gamma = 1***')
classifier_g = svm.SVC(gamma=1)
classifier_g.fit(train_data,train_label)
accuracy = classifier_g.predict(train_data)
print('\nTrain set accuracy : ' + str(100 * np.mean((accuracy == train_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(gamma=1)
classifier_g.fit(validation_data,validation_label)
accuracy = classifier_g.predict(validation_data)
print('\nValidation set accuracy : ' + str(100 * np.mean((accuracy == validation_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(gamma=1)
classifier_g.fit(test_data,test_label)
accuracy = classifier_g.predict(test_data)
print('\nTest set accuracy : ' + str(100 * np.mean((accuracy == test_label).astype(float))) + "%")
#print('--------------------------------------')

print('\n***SVM with RBF Kernel and Gamma = default***')
classifier_g = svm.SVC()
classifier_g.fit(train_data,train_label)
accuracy = classifier_g.predict(train_data)
print('\nTrain set accuracy : ' + str(100 * np.mean((accuracy == train_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC()
classifier_g.fit(validation_data,validation_label)
accuracy = classifier_g.predict(validation_data)
print('\nValidation set accuracy : ' + str(100 * np.mean((accuracy == validation_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC()
classifier_g.fit(test_data,test_label)
accuracy = classifier_g.predict(test_data)
print('\nTest set accuracy : ' + str(100 * np.mean((accuracy == test_label).astype(float))) + "%")
#print('--------------------------------------')

print('\nSVM with RBF Kernel, Gamma = default and varying C')
print('\nFor C = 1')
classifier_g = svm.SVC(C=1)
classifier_g.fit(train_data,train_label)
accuracy = classifier_g.predict(train_data)
print('\nTrain set accuracy : ' + str(100 * np.mean((accuracy == train_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(C=1)
classifier_g.fit(validation_data,validation_label)
accuracy = classifier_g.predict(validation_data)
print('\nValidation set accuracy : ' + str(100 * np.mean((accuracy == validation_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(C=1)
classifier_g.fit(test_data,test_label)
accuracy = classifier_g.predict(test_data)
print('\nTest set accuracy : ' + str(100 * np.mean((accuracy == test_label).astype(float))) + "%")
#print('--------------------------------------')

print('\nFor C = 10')
classifier_g = svm.SVC(C=10)
classifier_g.fit(train_data,train_label)
accuracy = classifier_g.predict(train_data)
print('\nTrain set accuracy : ' + str(100 * np.mean((accuracy == train_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(C=10)
classifier_g.fit(validation_data,validation_label)
accuracy = classifier_g.predict(validation_data)
print('\nValidation set accuracy : ' + str(100 * np.mean((accuracy == validation_label).astype(float))) + "%")
print('--------------------------------------')

classifier_g = svm.SVC(C=10)
classifier_g.fit(test_data,test_label)
accuracy = classifier_g.predict(test_data)
print('\nTest set accuracy : ' + str(100 * np.mean((accuracy == test_label).astype(float))) + "%")
#print('--------------------------------------')

print('\nFor C = 20')
classifier_g = svm.SVC(C=20)
classifier_g.fit(train_data,train_label)
accuracy = classifier_g.predict(train_data)
print('\nTrain set accuracy : ' + str(100 * np.mean((accuracy == train_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(C=20)
classifier_g.fit(validation_data,validation_label)
accuracy = classifier_g.predict(validation_data)
print('\nValidation set accuracy : ' + str(100 * np.mean((accuracy == validation_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(C=20)
classifier_g.fit(test_data,test_label)
accuracy = classifier_g.predict(test_data)
print('\nTest set accuracy : ' + str(100 * np.mean((accuracy == test_label).astype(float))) + "%")
#print('--------------------------------------')

print('\nFor C = 30')
classifier_g = svm.SVC(C=30)
classifier_g.fit(train_data,train_label)
accuracy = classifier_g.predict(train_data)
print('\nTrain set accuracy : ' + str(100 * np.mean((accuracy == train_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(C=30)
classifier_g.fit(validation_data,validation_label)
accuracy = classifier_g.predict(validation_data)
print('\nValidation set accuracy : ' + str(100 * np.mean((accuracy == validation_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(C=30)
classifier_g.fit(test_data,test_label)
accuracy = classifier_g.predict(test_data)
print('\nTest set accuracy : ' + str(100 * np.mean((accuracy == test_label).astype(float))) + "%")
#print('--------------------------------------')

print('\nFor C = 40')
classifier_g = svm.SVC(C=40)
classifier_g.fit(train_data,train_label)
accuracy = classifier_g.predict(train_data)
print('\nTrain set accuracy : ' + str(100 * np.mean((accuracy == train_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(C=40)
classifier_g.fit(validation_data,validation_label)
accuracy = classifier_g.predict(validation_data)
print('\nValidation set accuracy : ' + str(100 * np.mean((accuracy == validation_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(C=40)
classifier_g.fit(test_data,test_label)
accuracy = classifier_g.predict(test_data)
print('\nTest set accuracy : ' + str(100 * np.mean((accuracy == test_label).astype(float))) + "%")
#print('--------------------------------------')

print('\nFor C = 50')
classifier_g = svm.SVC(C=50)
classifier_g.fit(train_data,train_label)
accuracy = classifier_g.predict(train_data)
print('\nTrain set accuracy : ' + str(100 * np.mean((accuracy == train_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(C=50)
classifier_g.fit(validation_data,validation_label)
accuracy = classifier_g.predict(validation_data)
print('\nValidation set accuracy : ' + str(100 * np.mean((accuracy == validation_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(C=50)
classifier_g.fit(test_data,test_label)
accuracy = classifier_g.predict(test_data)
print('\nTest set accuracy : ' + str(100 * np.mean((accuracy == test_label).astype(float))) + "%")
#print('--------------------------------------')

print('\nFor C = 60')
classifier_g = svm.SVC(C=60)
classifier_g.fit(train_data,train_label)
accuracy = classifier_g.predict(train_data)
print('\nTrain set accuracy : ' + str(100 * np.mean((accuracy == train_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(C=60)
classifier_g.fit(validation_data,validation_label)
accuracy = classifier_g.predict(validation_data)
print('\nValidation set accuracy : ' + str(100 * np.mean((accuracy == validation_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(C=60)
classifier_g.fit(test_data,test_label)
accuracy = classifier_g.predict(test_data)
print('\nTest set accuracy : ' + str(100 * np.mean((accuracy == test_label).astype(float))) + "%")
#print('--------------------------------------')

print('\nFor C = 70')
classifier_g = svm.SVC(C=70)
classifier_g.fit(train_data,train_label)
accuracy = classifier_g.predict(train_data)
print('\nTrain set accuracy : ' + str(100 * np.mean((accuracy == train_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(C=70)
classifier_g.fit(validation_data,validation_label)
accuracy = classifier_g.predict(validation_data)
print('\nValidation set accuracy : ' + str(100 * np.mean((accuracy == validation_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(C=70)
classifier_g.fit(test_data,test_label)
accuracy = classifier_g.predict(test_data)
print('\nTest set accuracy : ' + str(100 * np.mean((accuracy == test_label).astype(float))) + "%")
#print('--------------------------------------')

print('\nFor C = 80')
classifier_g = svm.SVC(C=80)
classifier_g.fit(train_data,train_label)
accuracy = classifier_g.predict(train_data)
print('\nTrain set accuracy : ' + str(100 * np.mean((accuracy == train_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(C=80)
classifier_g.fit(validation_data,validation_label)
accuracy = classifier_g.predict(validation_data)
print('\nValidation set accuracy : ' + str(100 * np.mean((accuracy == validation_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(C=80)
classifier_g.fit(test_data,test_label)
accuracy = classifier_g.predict(test_data)
print('\nTest set accuracy : ' + str(100 * np.mean((accuracy == test_label).astype(float))) + "%")
#print('--------------------------------------')

print('\nFor C = 90')
classifier_g = svm.SVC(C=90)
classifier_g.fit(train_data,train_label)
accuracy = classifier_g.predict(train_data)
print('\nTrain set accuracy : ' + str(100 * np.mean((accuracy == train_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(C=90)
classifier_g.fit(validation_data,validation_label)
accuracy = classifier_g.predict(validation_data)
print('\nValidation set accuracy : ' + str(100 * np.mean((accuracy == validation_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(C=90)
classifier_g.fit(test_data,test_label)
accuracy = classifier_g.predict(test_data)
print('\nTest set accuracy : ' + str(100 * np.mean((accuracy == test_label).astype(float))) + "%")
#print('--------------------------------------')

print('\nFor C = 100')
classifier_g = svm.SVC(C=100)
classifier_g.fit(train_data,train_label)
accuracy = classifier_g.predict(train_data)
print('\nTrain set accuracy : ' + str(100 * np.mean((accuracy == train_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(C=100)
classifier_g.fit(validation_data,validation_label)
accuracy = classifier_g.predict(validation_data)
print('\nValidation set accuracy : ' + str(100 * np.mean((accuracy == validation_label).astype(float))) + "%")
#print('--------------------------------------')

classifier_g = svm.SVC(C=100)
classifier_g.fit(test_data,test_label)
accuracy = classifier_g.predict(test_data)
print('\nTest set accuracy : ' + str(100 * np.mean((accuracy == test_label).astype(float))) + "%")
#print('--------------------------------------')

##################
print('\n\n------------SVM ENDS-----------\n\n')
