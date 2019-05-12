#!/usr/bin/env python

# Global libs
import sys
import os
from argparse import ArgumentParser
import numpy as np
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt

#Add libsvm to current environment path
sd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(sd + '/libsvm-3.23/python/')
sys.path.append(sd + '/libsvm-3.23/windows/')
from svmutil import *  #requires dll!!!

def init_arg_parser(parents=[]):
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
		help='Give a set of parameters for each svm model. \n' +
			'See documentation of LIBSVM for more information \n' +
			'-t 4 => is our custom linear+RBF kernel',
		default=[
			'-t 0 -c 4 -b 1',
			'-t 1 -c 10 -g 1 -r 1 -d 1',
			'-t 2 -c 5 -g 0.5 -e 0.1',
			'-t 4 -c 4']
		)
	
	return parser

	
def arrange_subplots(pltc):
	cols = int(np.floor(np.sqrt(pltc)))
	rows = int(np.ceil(pltc/cols))
	return plt.subplots(rows,cols)
	
	
def plot_boundaries(Y, X, m, p, ax, h=1):
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
	k = float(Z.shape[1])
	
	c = plt.cm.get_cmap()
	for i,z in enumerate(Z.T):
		z = z.reshape(xx.shape)
		plt.axis('tight')
		ax.contour(xx, yy, z, linewidths=0.5, levels=0, colors=c(i/k))


def linearRBF(X):
	#https://stackoverflow.com/questions/10978261/libsvm-precomputed-kernels
	s = X.shape
	labels = np.arange(1,s[0]+1)[:,np.newaxis]
	X = np.dot(X, X.T)
	X = np.c_[labels, X]
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
	axes = axes.flatten()
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
			prob = svm_problem(Y, linearRBF(X), isKernel=True) #Apply linear kernel to data.
		else:
			prob = svm_problem(Y, X)
			
		m = svm_train(prob, param)
		
		#Get result
		SV = m.get_SV()
		SV = np.array([[v[1], v[2]] for v in SV])
		
		#Show result
		plot_boundaries(Y, X, m, param, ax) #Bounds
		ax.scatter(X[:,0], X[:,1], s=1, c=Y) #GT
		ax.scatter(SV[:,0], SV[:,1], s=3, c='r', marker='v') #SV
		plt.show(block=False)
		plt.pause(0.1)

	plt.show()
	
	
