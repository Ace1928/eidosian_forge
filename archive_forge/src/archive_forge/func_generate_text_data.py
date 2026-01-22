import inspect
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import autokeras as ak
def generate_text_data(num_instances=100):
    vocab = np.array([['adorable', 'clueless', 'dirty', 'odd', 'stupid'], ['puppy', 'car', 'rabbit', 'girl', 'monkey'], ['runs', 'hits', 'jumps', 'drives', 'barfs'], ['crazily.', 'dutifully.', 'foolishly.', 'merrily.', 'occasionally.']])
    return np.array([' '.join([vocab[j][np.random.randint(0, 5)] for j in range(4)]) for i in range(num_instances)])