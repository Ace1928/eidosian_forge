import tensorflow as tf
from tensorflow.experimental import numpy as tfnp
from keras.src.backend.common import standardize_dtype
from keras.src.backend.config import floatx
from keras.src.random.seed_generator import SeedGenerator
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed
def tf_draw_seed(seed):
    return tf.cast(draw_seed(seed), dtype='int32')