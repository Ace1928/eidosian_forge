import math
from keras_tuner.src import protos
def prob_to_index(prob, n_index):
    """Convert cumulative probability to 0-based index in the given range."""
    ele_prob = 1 / n_index
    index = int(math.floor(prob / ele_prob))
    if index == n_index:
        index -= 1
    return index