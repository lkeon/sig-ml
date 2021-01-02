# sig-ml
# Signal analysis with machine learning
# Leon Kocjancic
# leon.567@gmail.com

import numpy as np

class NeuralNet():
	"""Class for instantiating neural network models."""
	
	def __init__(self):
		self.layers = {} # Dictionary containing data of all layers
		self.layerNo = 0 # Number of all layers, input and output layers
		self.x_train = None
		self.y_train = None
		self.optimizer = None
		self.loss = None

	def add(self, nodes, layer_type='dense', activation='relu', input_shape=None):
		'''
		Add a layer to the neural network and constructs a dictionary of layers containing
		all the parameters, hyperarameters and data. Input shape is a tuple, with first argument
		being the number of features and the second number of samples.
		'''
		self.layerNo += 1
		self.layers[self.layerNo] = {}
		self.layers[self.layerNo]['nodes'] = nodes
		self.layers[self.layerNo]['type'] = layer_type
		self.layers[self.layerNo]['activation'] = activation

		if self.layerNo == 1:
			if input_shape == None:
				print('Add shape argument to the first layer. input_shape=(num_features, num_samples).')
			
			else:
				self.featureNo = input_shape[0]
				self.sampleNo = input_shape[1]


	def compile(self, loss='binary_crossentropy', optimizer='gradient_descent'):
		'''
		Compiles the added layers and creates random or zero value matrix placeholders
		for training process.
		'''
		self.loss = loss
		self.optimizer = optimizer

		# Add last layer corresponding to the cost function
		if loss == 'binary_crossentropy':
			self.layerNo += 1
			self.layers[self.layerNo]['nodes'] = 1
			self.layers[self.layerNo]['type'] = 'dense'
			self.layers[self.layerNo]['activation'] = 'logistic'

		else:
			print('Loss function not recognized.')

		# Initialize matrices
		prevNodesNo = self.featureNo

		for layer in range(1, self.layerNo + 1):

			nodesNo = self.layers[layer]['nodes']
			
			self.layers[layer]['W'] = np.random.randn(nodesNo, prevNodesNo) * 0.01 # initalize randomly to enforce assimetry, multiply by a small number to be closer to hig grad
			self.layers[layer]['b'] = np.zeros((nodesNo, 1))

			prevNodesNo = nodesNo



	def fit(self, x_train, y_train, epochs=10):
		pass

	def _forward_propagation(self):
		pass

	def _back_propagation(self):
		pass

	def _compute_cost(self):
		pass

	@staticmethod
	def sigmoid(x):
		'''
		Calculates sigmoid function, where x is a number, vector or matrix.
		'''
		y = 1 / (1 + np.exp(-x))

		return x
