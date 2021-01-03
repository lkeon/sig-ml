# sig-ml
# Signal analysis with machine learning
# Leon Kocjancic
# leon.567@gmail.com
#
# Example of usage

from sigml import NeuralNet
import numpy as np

model = NeuralNet()
model.add(10, layer_type='dense', activation='relu', input_shape=(20, 500))
model.add(20, layer_type='dense', activation='relu')
model.add(25, layer_type='dense', activation='relu')
model.add(10, layer_type='dense', activation='relu')
model.add(10, layer_type='dense', activation='relu')
model.compile(loss='binary_crossentropy', optimizer='gradient_descent')
