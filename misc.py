import numpy as np
import matplotlib.pyplot as plt


def subset(X, Y, samples=100):
	i = np.random.choice(range(len(Y)), samples)
	return X[i,:], Y[i], i


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
