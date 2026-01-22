import functools
import jax.experimental.sparse as jax_sparse
import jax.numpy as jnp
from keras.src.utils import jax_utils
def wrap_elementwise_unary(func):

    @functools.wraps(func)
    def sparse_wrapper(x, *args, **kwargs):
        if isinstance(x, jax_sparse.BCOO):
            if not linear and (not x.unique_indices):
                x = jax_sparse.bcoo_sum_duplicates(x)
            return jax_sparse.BCOO((func(x.data, *args, **kwargs), x.indices), shape=x.shape)
        else:
            return func(x, *args, **kwargs)
    return sparse_wrapper