#!/usr/bin/env python

# Global libs
from sys import stdout
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

# Local libs
from libsvm import *
from misc import *


def init_arg_parser(parents=[]):
	'''
	Initialize an ArgumentParser for this script.
	
	Args:
		parents: A list of ArgumentParsers of other scripts, if there are any.
		
	Returns:
		parser: The ArgumentParsers.
	'''
	parser = ArgumentParser(
		description='Demo for Classifying Handwritten Digits 0-4 via SVM',
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
		default=[0.01, 2, 8]
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
		default=[0.125, 0.25, 0.75]
		)
	
	parser.add_argument(
		'--coef', '-r',
		type=float,
		nargs='*',
		help='Give a set of coef0 for grid search. ' +
			'See LIBSVM doc for more information. ',
		default=[1, 2]
		)
	
	return parser


if __name__ == '__main__':

	knames = [
		"Linear",
		"Polynomial",
		"RBF",
		"Sigmoid",
		"Linear+RBF"
		]
	
	report = {
		0:np.array([0]),
		1:np.array([0]),
		2:np.array([0]),
		3:np.array([0]),
		4:np.array([0])
		}

	def viz_grid_search(param_str, acc):
		p = svm_parameter(param_str)
		report[p.kernel_type] = np.append(report[p.kernel_type], acc)
		print("for {}".format(param_str))
		sys.stdout.flush()

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
	tX = np.genfromtxt(args.train_data, delimiter=',')
	tY = np.genfromtxt(args.train_labels, delimiter=',')

	print("Find best params on subset...")
	x, y, _ = subset(tX, tY, 100)
	param, grid_acc, param_str = grid_search(x, y, param_dict, viz_grid_search)

	print("Train svm model with full set...")
	stdout.flush()
	prob = svm_problem(tY, tX)
	m = svm_train(prob, param)
	
	print("Load evaluation data...")
	eX = np.genfromtxt(args.eval_data, delimiter=',')
	eY = np.genfromtxt(args.eval_labels, delimiter=',')

	print("Evaluate svm model...")
	stdout.flush()
	_, pred_acc, _ = svm_predict(eY, eX, m)
	print("Parameters: {}".format(param_str))
	print("Predicted accuracy with evaluation dataset:\n    {}".format(pred_acc))
	
	print("Generate linear+RBF kernels")
	param = svm_parameter("-t 4 -c 4 -g 0.5 -h 0")
	prob = svm_problem(tY, linearRBF(tX,tX, g=param.gamma), isKernel=True)
	eK = linearRBF(eX,tX, g=param.gamma)
	
	print("Train linear+RBF...")
	stdout.flush()
	m = svm_train(prob, param)
	
	print("Evaluate linear+RBF model...")
	stdout.flush()
	_, pred_acc, _ = svm_predict(eY, eK, m)
	print("Predicted accuracy with evaluation dataset:\n    {}".format(pred_acc))
	
	print("\nCompare kernels:")
	for k,v in report.items():
		print("{} Avg-Acc:\n \t {}".format(knames[k], np.mean(v)))
