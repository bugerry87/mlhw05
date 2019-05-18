#!/usr/bin/env python

# Global libs
from sys import stdout
from argparse import ArgumentParser
import numpy as np
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt

# Local libs
from libsvm import *


def init_arg_parser(parents=[]):
	'''
	Initialize an ArgumentParser for this script.
	
	Args:
		parents: A list of ArgumentParsers of other scripts, if there are any.
		
	Returns:
		parser: The ArgumentParsers.
	'''
	parser = ArgumentParser(
		description='Demo Digits of ML_HW05',
		parents=parents
		)
	
	parser.add_argument(
		'--train_data', '-X',
		help='The filename of a csv with training data.',
		default='X_train.csv'
		)

	parser.add_argument(
		'--train_labels', '-Y',
		help='The filename of a csv with training labels.',
		default='Y_train.csv'
		)
	
	parser.add_argument(
		'--eval_data', '-x',
		help='The filename of a csv with test data.',
		default='X_test.csv'
		)

	parser.add_argument(
		'--eval_labels', '-y',
		help='The filename of a csv with test labels.',
		default='Y_test.csv'
		)
	
	parser.add_argument(
		'--kernel_type', '-t',
		type=int,
		nargs='*',
		help='Give a set of kernel types for grid search. ' +
			'See LIBSVM doc for more information. '
			'-t 4 => is our custom linear+RBF kernel',
		default=[0, 1, 2]
		)
	
	parser.add_argument(
		'--cost', '-c',
		type=float,
		nargs='*',
		help='Give a set of costs for grid search. ' +
			'See LIBSVM doc for more information. '
			'-t 4 => is our custom linear+RBF kernel',
		default=[1, 2, 4, 8]
		)
	
	parser.add_argument(
		'--degree', '-d',
		type=float,
		nargs='*',
		help='Give a set of degrees for grid search. ' +
			'See LIBSVM doc for more information. ',
		default=[2, 3, 4]
		)
	
	parser.add_argument(
		'--gamma', '-g',
		type=float,
		nargs='*',
		help='Give a set of degrees for grid search. ' +
			'See LIBSVM doc for more information. ',
		default=[0.125, 0.25, 0.5, 0.75]
		)
	
	parser.add_argument(
		'--coef', '-r',
		type=float,
		nargs='*',
		help='Give a set of coef0 for grid search. ' +
			'See LIBSVM doc for more information. ',
		default=[0, 1, 2]
		)
	
	return parser


def viz_grid_search(param_str, acc):
	print("for {}".format(param_str))
	sys.stdout.flush()


def subset(X, Y, samples=100):
	i = np.random.choice(range(len(Y)), samples)
	return X[i,:], Y[i], i

if __name__ == '__main__':
	print("Parse input arguments...")
	parser = init_arg_parser()
	args, _ = parser.parse_known_args()
	param_dict = {
		'-t':args.kernel_type,
		'-c':args.cost,
		'-d':args.degree,
		'-g':args.gamma,
		'-r':args.coef
	}

	print("Load training data...")
	X = np.genfromtxt(args.train_data, delimiter=',')
	Y = np.genfromtxt(args.train_labels, delimiter=',')
	RBFprob = svm_problem(Y, linearRBF(X,X, g=0.5), isKernel=True)

	print("Find best params on subset...")
	x, y, _ = subset(X, Y, 100)
	param, grid_acc, param_str = grid_search(x, y, param_dict, viz_grid_search)

	print("Train svm model with full set...")
	stdout.flush()
	prob = svm_problem(Y, X)
	m = svm_train(prob, param)
	
	print("Load evaluation data...")
	X = np.genfromtxt(args.eval_data, delimiter=',')
	Y = np.genfromtxt(args.eval_labels, delimiter=',')

	print("Evaluate svm model...")
	stdout.flush()
	_, pred_acc, _ = svm_predict(Y, X, m)
	print("Parameters: {}".format(param_str))
	print("Predicted accuracy with evaluation dataset:\n    {}".format(pred_acc))
	
	print("Train linear+RBF...")
	stdout.flush()
	param = svm_parameter("-t 4 -c 4 -h 0")
	m = svm_train(RBFprob, param)
	
	print("Evaluate linear+RBF model...")
	stdout.flush()
	_, pred_acc, _ = svm_predict(Y, linearRBF(X,X, 0.5), m)
	print("Predicted accuracy with evaluation dataset:\n    {}".format(pred_acc))
	
	
