# sig-ml
# Signal analysis with machine learning
# Leon Kocjancic
# leon.567@gmail.com

import numpy as np
import warnings as wr


class NeuralNet():
    """Class for instantiating neural network models."""

    def __init__(self):
        self.layers = 0  # Number of all layers, input and output layers
        self.neurons = []  # List of neurons in each layer
        self.activations = []  # List of activation functions in each layer
        self.x_train = None
        self.y_train = None
        self.optimizer = None
        self.loss = None

    def add(self, neurons_num,
            layer_type='dense',
            activation='relu',
            input_shape=None):
        '''Add a layer to the neural network and constructs a dictionary of
        layers containing all the parameters, hyperarameters and data.
        Input shape is a tuple, with first argument being the number of
        features and the second number of samples.
        '''
        self.layers += 1
        self.neurons.append(neurons_num)
        self.activations.append(activation)
        if self.layers == 1:
            if input_shape is None:
                wr.warn('Input shape variable not defined.')
            else:
                self.featureNo = input_shape[0]
                self.sampleNo = input_shape[1]

    def compile(self, loss='binary_crossentropy',
                optimizer='gradient_descent'):
        '''Compiles the added layers and creates random or zero value
        matrix placeholders for training process.
        '''
        self.loss = loss
        self.optimizer = optimizer

        if loss == 'binary_crossentropy':
            self.layers += 1
            self.neurons.append(1)
            self.activations.append('sigmoid')
        else:
            raise ValueError('Loss function not recognized.')

        for layer in range(1, self.layers + 1):
            nodesNo = self.layers[layer]['nodes']
            # initalize randomly to enforce assimetry
            # multiply to a small num to increase grad
            self.layers[layer]['W'] = \
                np.random.randn(nodesNo, prevNodesNo) * 0.01
            self.layers[layer]['b'] = np.zeros((nodesNo, 1))
            prevNodesNo = nodesNo

    def _forward_propagation(self):
        '''Implements forward propagation throuhout all layers.
        '''
        for layer in range(1, self.layerNo + 1):
            W = self.layers[layer]['W']
            b = self.layers[layer]['b']
            z = np.dot(W, self.x_train) + b
            # Calculate activations according to the specified fcn
            activationFcn = elf.layers[layer]['activation']
            if activationFcn == 'relu':
                a = self.relu(z)
                self.layers[layer]['a'] = a
            elif activationFcn == 'sigmoid':
                a = self.sigmoid(z)
                self.layers[layer]['a'] = a
            elif activationFcn == 'tanh':
                a = np.tanh(z)
                self.layers[layer]['a'] = a
            else:
                txt = 'Activation functoin {} in layer {} not valid.' \
                    .format(activationFcn, layer)
                print(txt)

    def _back_propagation(self):
        '''Implements back propagation through all layers.
        '''
        if self.loss == 'binary_crossentropy':
            pass

    def _update_params(self):
        pass

    def _compute_cost(self):
        '''
        Compute cost according to the specified cost function.
        '''
        a = self.layers[self.layerNo]['a']
        y = self.y_train
        m = y.shape[1]
        if self.loss == 'binary_crossentropy':
            log_probabilities = np.multiply(y, np.log(a)) + \
                np.multiply((1 - y), np.log(1 - a))
            cost = -1/m * np.sum(log_probabilities)
        else:
            txt = 'Loss function {} not valid.'.format(self.loss)
            print(txt)
        return cost

    def _compute_cost_deriv(self):
        '''
        Compute the derivative of cost function.
        '''
        if self.loss

    def fit(self, x_train, y_train, epochs=10):
        pass

    @staticmethod
    def sigmoid(x):
        '''Calculates sigmoid activation function, where x is a number,
        vector or matrix.
        '''
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_deriv(x):
        '''
        Calcuates derivative of the sigmoid function on a number or a vector.
        '''
        return NeuralNet.sigmoid(x) * (1 - NeuralNet.sigmoid(x))

    @staticmethod
    def relu(x):
        '''
        Calculates rectified linear activation function.
        '''
        return x[x <= 0] = 0

    @staticmethod
    def relu_deriv(x):
        '''Calculate derivative of the rectified linear function on a scalar
        or vector.
        '''
        x = np.array(x)
        out = ones(x.shape)
        out[x <= 0] = 0
        return out
