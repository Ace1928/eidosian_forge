import inspect
import jax
import jax.lax as lax
import jax.numpy as jnp
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
def handle_cumulative_probs(logprobs_k, scores_k):
    timestamp_logprob = jax.nn.logsumexp(logprobs_k[self.timestamp_begin:], axis=-1)
    max_text_token_logprob = jnp.max(logprobs_k[:self.timestamp_begin])
    return jnp.where(timestamp_logprob > max_text_token_logprob, scores_k.at[:self.timestamp_begin].set(-float('inf')), scores_k)