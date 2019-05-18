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
from kernels import linearRBF

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
			'-t 4 -c 4 -h 0 -g 0.5'
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


def prediction(X, Y, m, p):
	options = '-b {}'.format(p.probability)
	if p.kernel_type == 4:
		K = linearRBF(X,X, p.gamma)
	else:
		K = X
	
	# Put the result into a color plot
	[y, acc, z] = svm_predict(Y, K, m, options)
	y = np.array(y)
	z = np.array(z)
	return y, acc, z

	
def plot_boundaries(X, m, p, ax, h=1):
	#http://scikit-learn.sourceforge.net/0.5/auto_examples/svm/plot_iris.html	
	# create a mesh to plot in
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	X = np.c_[xx.ravel(), yy.ravel()]
	
	# Put the result into a color plot
	_, _, Z = prediction(X, [], m, p)
	
	k = Z.shape[1]
	c = plt.cm.get_cmap()
	for i,z in enumerate(Z.T):
		z = z.reshape(xx.shape)
		plt.axis('tight')
		cmap = np.tile(c(i**2/k), (10,1)) #keep some colors in backup...
		ax.contour(xx, yy, z, linewidths=0.5, levels=0, colors=cmap)


def plot_prediction(X, Y, m, ax):
	# Put the result into a color plot
	i = np.array(m.get_sv_indices()) - 1
	ax.scatter(X[:,0], X[:,1], s=1, c=Y) #Prediction
	ax.scatter(X[i,0], X[i,1], s=3, c='r', marker='v') #SV


if __name__ == '__main__':
	#Parse input arguments
	parser = init_arg_parser()
	args = parser.parse_args()
	
	#Load data
	X = np.genfromtxt(args.data, delimiter=',')
	Y = np.genfromtxt(args.labels, delimiter=',')
	
	#Plot GT
	pltc = len(args.svm_params)
	_, axes = arrange_subplots(pltc)
	axes = axes.flatten() #flatten for easier usage.
	
	for i in range(0,pltc):
		ax = axes[i]
		ax.scatter(X[:,0], X[:,1], s=1, c='gray')
	plt.show(block=False)
	plt.pause(0.1)
	
	titles = ("Linear", "Polynomial", "RBF", "Sigmoid", "linear+RBF")
	
	#Train svm models
	for i in range(0,pltc):
		ax = axes[i]
		param = svm_parameter(args.svm_params[i])
		ax.title.set_text(titles[param.kernel_type])
		
		prob = None
		if param.kernel_type == 4: #That's our custom linear+RBF kernel type.
			prob = svm_problem(Y, linearRBF(X,X, param.gamma), isKernel=True) #Apply linearRBF kernel to data.
		else:
			prob = svm_problem(Y, X)
		
		print("\n" + titles[param.kernel_type] + "\n")
		stdout.flush()
		m = svm_train(prob, param)
		
		#Get result
		y, acc, _ = prediction(X, Y, m, param)
		
		#Show result
		if param.kernel_type != 4:
			plot_boundaries(X, m, param, ax) #Bounds
		plot_prediction(X, y, m, ax) #Prediction
		plt.show(block=False)
		plt.pause(0.1)

	plt.show() #Show and stay
	
	
