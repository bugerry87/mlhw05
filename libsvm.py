'''Import helper for 'any' LibSVM version.'''
import sys
import os
from glob import iglob
import numpy as np

# Add libsvm to current environment path
sd = os.path.dirname(os.path.realpath(__file__))
for p in iglob(sd + '/libsvm*/python/'):
	print("Append to PATH: " + p)
	sys.path.append(p)
	break

# add tools
for p in iglob(sd + '/libsvm*/tools/'):
	print("Append to PATH: " + p)
	sys.path.append(p)
	break


from svmutil import *


def linearRBF(u, v, g=0.01):
	rbf = np.sum(u**2, axis=1)[:,None] - np.sum(v**2, axis=1)[None,:]
	rbf = np.exp(-g * rbf**2)
	X = np.dot(u, v.T) + rbf
	
	index = np.arange(1,X.shape[0]+1)[:,np.newaxis]
	X = np.c_[index, X]
	X = [list(row) for row in X]
	return X


def prediction(X, Y, m, p, isKernel=False):
	options = '-b {}'.format(p.probability)
	
	if isKernel:
		K = X
	elif p.kernel_type == 4:
		K = linearRBF(X,X, p.gamma)
	else:
		K = X
	
	# Put the result into a color plot
	[y, acc, z] = svm_predict(Y, K, m, options)
	y = np.array(y)
	z = np.array(z)
	return y, acc, z


def grid_search(X, Y, param_dict, viz_func=None):	
	best_acc = 0.0
	keys = list(param_dict)
	param_pattern = " {} ".join(keys) + " {}"
	
	grid = []
	def gen_grid(i, value_list, value=None):
		if value != None:
			value_list.append(value)
		
		if i < len(keys):
			k = keys[i]
			values = param_dict[k]
			for v in values:
				if k == '-t' and v == 4:
					continue #Sry but we skip custom kernels. They take too much time.
				gen_grid(i+1, value_list.copy(), v)
		else:
			grid.append(value_list)
		pass
	gen_grid(0, [])
	
	for row in grid:
		param_str = param_pattern.format(*row)
		acc = svm_train(Y, X, param_str + " -v 3 -q")
		
		if viz_func != None:
				viz_func(param_str, acc)
		
		if acc > best_acc:
			best_acc = acc
			best_param_str = param_str
	
	return svm_parameter(best_param_str), best_acc, best_param_str
		
