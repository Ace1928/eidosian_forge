import math
from keras_tuner.src import protos
def index_to_prob(index, n_index):
    """Convert 0-based index in the given range to cumulative probability."""
    ele_prob = 1 / n_index
    return (index + 0.5) * ele_prob