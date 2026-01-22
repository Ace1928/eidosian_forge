from functools import partial
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.typing import ResultBatch
from ..jacobian_products import _compute_jvps
from .jax import _NonPytreeWrapper
def pure_callback_wrapper(p):
    new_tapes = _set_fn(tapes.vals, p)
    res = tuple(_to_jax(execute_fn(new_tapes)))
    return jax.tree_map(lambda r, s: r.T if r.ndim > s.ndim else r, res, shape_dtype_structs)