import os
import random
import string
import time
import numpy as np
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.layers.preprocessing import index_lookup
def tensor_gen(batch, num_elements):
    data = []
    for _ in range(batch):
        batch_element = []
        for _ in range(num_elements - 1):
            tok = ''.join((random.choice(string.ascii_letters) for i in range(2)))
            batch_element.append(tok)
        batch_element.append('')
        data.append(batch_element)
    return tf.constant(data)