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
		self.data = None
		self.optimizer = None
		self.lossFunction = None

	def add_layer(nodesNo, layer_type='dense', activation='relu'):
		self.layerNo += 1
		self.layers[self.layerNo] = {}
		self.layers[self.layerNo]['type'] = layer_type
		self.layers[self.layerNo][activation] = activation

	def compile_layers(loss='binary_crossentropy', optimizer='gradient'):
		self.lossFunction = loss
		self.optimizer = optimizer
