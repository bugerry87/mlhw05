import numpy as np

def linearRBF(u, v, g=0.01):
	linear = np.dot(u, v.T)
	rbf = np.sum(u**2, axis=1)[:,None] + np.sum(v**2, axis=1)[None,:] - 2*linear
	rbf = np.abs(rbf) * -g
	rbf = np.exp(rbf)
	X = linear + rbf
	
	index = np.arange(1,X.shape[0]+1)[:,np.newaxis]
	X = np.c_[index, X]
	X = [list(row) for row in X]
	return X