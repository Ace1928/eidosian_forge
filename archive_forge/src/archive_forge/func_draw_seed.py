import random as python_random
import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
from keras.src.utils import jax_utils
def draw_seed(seed):
    from keras.src.backend import convert_to_tensor
    if isinstance(seed, SeedGenerator):
        return seed.next()
    elif isinstance(seed, int):
        return convert_to_tensor([seed, 0], dtype='uint32')
    elif seed is None:
        return global_seed_generator().next(ordered=False)
    raise ValueError(f'Argument `seed` must be either an integer or an instance of `SeedGenerator`. Received: seed={seed} (of type {type(seed)})')