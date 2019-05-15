#!/usr/bin/env python

# Global libs
from sys import stdout
from argparse import ArgumentParser
import numpy as np
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt

# Local libs
import libsvm
from svmutil import *

def init_arg_parser(parents=[]):
	'''
	Initialize an ArgumentParser for this script.
	
	Args:
		parents: A list of ArgumentParsers of other scripts, if there are any.
		
	Returns:
		parser: The ArgumentParsers.
	'''
	parser = ArgumentParser(
		description='Demo SVM of ML_HW05',
		parents=parents
		)
	
	parser.add_argument(
		'--data', '-x',
		help='The filename of a csv with datapoints.',
		default='Plot_X.csv'
		)

	parser.add_argument(
		'--labels', '-y',
		help='The filename of a csv with labels.',
		default='Plot_Y.csv'
		)
	
	parser.add_argument(
		'--svm_params', '-p',
		nargs='+',
		help='Give a set of parameters for each svm model. ' +
			'See documentation of LIBSVM for more information. ' +
			'-t 4 => is our custom linear+RBF kernel',
		default=[
			'-t 0 -c 4 -b 1',
			'-t 1 -c 10 -g 1 -r 1 -d 2',
			'-t 2 -c 5 -g 0.5 -e 0.1',
			'-t 4 -c 4'
			]
		)
	
	return parser

	
def arrange_subplots(pltc):
	'''
	Arranges a given number of plots to well formated subplots.
	
	Args:
		pltc: The number of plots.
	
	Returns:
		fig: The figure.
		axes: A list of axes of each subplot.
	'''
	cols = int(np.floor(np.sqrt(pltc)))
	rows = int(np.ceil(pltc/cols))
	fig, axes = plt.subplots(cols,rows)
	if not isinstance(axes, np.ndarray):
		axes = np.array([axes]) #fix format so it can be used consistently.
	
	return fig, axes
	
	
def plot_boundaries(X, m, p, ax, h=1):
	#http://scikit-learn.sourceforge.net/0.5/auto_examples/svm/plot_iris.html
	# create a mesh to plot in
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	X = np.c_[xx.ravel(), yy.ravel()]
	options = '-b {}'.format(param.probability)
	
	if p.kernel_type == 4:
		X = linearRBF(X)
	
	# Put the result into a color plot
	[_, _, Z] = svm_predict([], X, m, options)
	Z = np.array(Z)
	
	k = Z.shape[1]
	c = plt.cm.get_cmap()
	
	for i,z in enumerate(Z.T):
		z = z.reshape(xx.shape)
		plt.axis('tight')
		cmap = np.tile(c(i**2/k), (10,1)) #keep some colors in backup...
		ax.contour(xx, yy, z, linewidths=0.5, levels=0, colors=cmap)


def linearRBF(X):
	'''
	
	Returns:
		X: The Gram Matrix.
	'''

	#https://stackoverflow.com/questions/10978261/libsvm-precomputed-kernels
	#https://stackoverflow.com/questions/15556116/implementing-support-vector-machine-efficiently-computing-gram-matrix-k
	s = X.shape
	index = np.arange(1,s[0]+1)[:,np.newaxis]
	#norms = (X**2).sum(axis=1)
	X = np.dot(X, X.T)
	#X *= -2
	#X += norms.reshape(-1,1)
	#X += norms
	
	#X *= -0.1**2 / 2
	#X = np.exp(X,X)
	
	X = np.c_[index, X]
	X = [list(row) for row in X]
	#TODO: make linear kernel to work, then add RBF
	return X
	

if __name__ == '__main__':
	#Parse input arguments
	parser = init_arg_parser()
	args = parser.parse_args()
	
	#Load data
	X = np.genfromtxt(args.data, delimiter=',')
	Y = np.genfromtxt(args.labels, delimiter=',')
	
	#Plot GT
	pltc = len(args.svm_params)
	fig, axes = arrange_subplots(pltc)
	axes = axes.flatten() #flatten for easier usage.
	
	for i in range(0,pltc):
		ax = axes[i]
		ax.scatter(X[:,0], X[:,1], s=1, c=Y)
	plt.show(block=False)
	plt.pause(0.1)
	
	#Train svm models
	for i in range(0,pltc):
		ax = axes[i]
		param = svm_parameter(args.svm_params[i])
		prob = None
		
		if param.kernel_type == 4: #That's our custom linear+RBF kernel type.
			prob = svm_problem(Y, linearRBF(X), isKernel=True) #Apply linearRBF kernel to data.
		else:
			prob = svm_problem(Y, X)
		
		stdout.flush()
		m = svm_train(prob, param)
		
		#Get result
		SV = m.get_SV()
		SV = np.array([[v[1], v[2]] for v in SV])
		
		#Show result
		plot_boundaries(X, m, param, ax) #Bounds
		ax.scatter(X[:,0], X[:,1], s=1, c=Y) #GT
		ax.scatter(SV[:,0], SV[:,1], s=3, c='r', marker='v') #SV
		plt.show(block=False)
		plt.pause(0.1)

	plt.show() #Show and stay
	
	
