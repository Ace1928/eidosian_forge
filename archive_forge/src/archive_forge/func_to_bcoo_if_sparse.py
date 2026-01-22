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
def to_bcoo_if_sparse(x, maybe_symbolic_x):
    if isinstance(maybe_symbolic_x, KerasTensor) and maybe_symbolic_x.sparse:
        return jax_sparse.BCOO.fromdense(x, nse=1)
    return x