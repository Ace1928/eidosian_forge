import numpy as np
import math
from Algorithm import Algorithm
from Constants import USER_SEED
def feedforward(self, input_matrix):
    input_matrix = np.array(input_matrix)
    for b, w in zip(self.biases, self.weights):
        input_matrix = tanh(np.dot(w, input_matrix) + b)
    return input_matrix