import inspect
import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import sparse
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
def body_fun(i, val):
    b = i % batch_size
    pos = i // batch_size
    return val.at[i].set(jnp.array([b] + [jnp.array(input_ids)[b, pos + j] for j in range(self.ngram_size)]))