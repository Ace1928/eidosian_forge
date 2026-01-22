import inspect
import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import sparse
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
def get_banned_tokens_mask(self, latest_tokens: jnp.ndarray, previous_ngrams) -> jnp.ndarray:
    """
        Determines which tokens must be banned given latest tokens and the previously seen
        ngrams.
        """

    @sparse.sparsify
    @jax.vmap
    def inner_fn(latest_tokens, previous_ngrams):
        return previous_ngrams[tuple(latest_tokens)]
    return sparse.bcoo_todense(inner_fn(latest_tokens, previous_ngrams))