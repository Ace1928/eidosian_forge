import inspect
import jax
import jax.lax as lax
import jax.numpy as jnp
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
def handle_pairs(input_ids_k, scores_k):
    last_was_timestamp = jnp.where(cur_len - self.begin_index >= 1, True, False)
    last_was_timestamp = jnp.where(input_ids_k[cur_len - 1] >= self.timestamp_begin, True and last_was_timestamp, False)
    penultimate_was_timestamp = jnp.where(cur_len - self.begin_index < 2, True, False)
    penultimate_was_timestamp = jnp.where(input_ids_k[cur_len - 2] >= self.timestamp_begin, True, penultimate_was_timestamp)
    return jnp.where(last_was_timestamp, jnp.where(penultimate_was_timestamp > 0, scores_k.at[self.timestamp_begin:].set(-float('inf')), scores_k.at[:self.eos_token_id].set(-float('inf'))), scores_k)