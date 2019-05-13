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
from grid import find_parameters


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
		'--subset', '-z',
		help='Subset for parameter grid search.',
		default='MNIST0-4_YX_sub100.csv'
		)
	
	return parser
	

if __name__ == '__main__':
	print("Parse input arguments...")
	parser = init_arg_parser()
	args, options = parser.parse_known_args()

	print("Load data...")
	TX = np.genfromtxt(args.train_data, delimiter=',')
	TY = np.genfromtxt(args.train_labels, delimiter=',')
	EX = np.genfromtxt(args.eval_data, delimiter=',')
	EY = np.genfromtxt(args.eval_labels, delimiter=',')

	print("Find best params via grid search from libsvm...")
	_, best_params = find_parameters(args.subset, options)

	print("Train svm model...")
	param = svm_parameter()
	param.kernel_type = 3
	param.C = best_params['c']
	param.gamma = best_params['g']
	stdout.flush()
	prob = svm_problem(TY, TX)
	m = svm_train(prob, param)

	print("Evaluate svm model...")
	_, acc, _ = svm_predict(EY, EX, m)
	print(acc)
	
	
