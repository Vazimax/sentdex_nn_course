import numpy as np

np.random.seed(0)


X = [   [1,2,3,4],
        [7,4,-3,0.65],
        [8.7,0.26,3,-4],
]

class LayerDense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self,inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = LayerDense(4,2)
layer2 = LayerDense(2,1)

layer1.forward(X)
layer2.forward(layer1.output)
