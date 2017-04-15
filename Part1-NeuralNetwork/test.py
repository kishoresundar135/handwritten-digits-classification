import numpy as np
from scipy.io import loadmat
mat = loadmat('D:\Work\Machine Learning\PA1\mnist_all.mat')
#print mat
array = np.array([])
array = np.append(array,0)
array = np.append(array,1)
array = np.append(array,7)
array = np.append(array,3)
array = np.append(array,4)
array = np.append(array,5)
array = np.append(array,6)
print array
#array.resize(2,3)
#print array

for i in range(0,array.size):
    print array[i]

index = np.argmax(array)
print index
    