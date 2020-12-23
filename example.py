#import the datasets
# from datasets import *
from load import *
import numpy as np
import sys

if __name__ == '__main__':

	dataset = sys.argv[1]
	# load the data
	X, origY = eval('load_' + load + '(return_X_y=True)')
	n, d = X.shape

	# map the values of Y from 0 to r-1
	domY= np.unique(origY)
	r= len(domY)
	Y= np.zeros(X.shape[0], dtype= np.int)
	for i,y in enumerate(domY):
		Y[origY==y]= i

	print('\nThe shape of the dataset is : ' + str(n) + ' x ' + str(d))

	print('\nNumber of classes are : ', r)

	print('\nThe dataset : ', )
	print(X)

	print('\nThe labels : ')
	print(Y)