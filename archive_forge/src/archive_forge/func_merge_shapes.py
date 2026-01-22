import types
import jax
import jax.experimental.sparse as jax_sparse
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import tree
from jax.tree_util import Partial
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.jax import distribution_lib
def merge_shapes(shape1, shape2):
    return tuple([d1 if d1 == d2 else None for d1, d2 in zip(shape1, shape2)])