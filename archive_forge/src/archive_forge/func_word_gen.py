import itertools
import random
import string
import time
import numpy as np
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.layers.preprocessing import hashing
def word_gen():
    for _ in itertools.count(1):
        yield ''.join((random.choice(string.ascii_letters) for i in range(2)))