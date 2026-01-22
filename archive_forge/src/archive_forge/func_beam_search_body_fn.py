import copy
import inspect
import warnings
from functools import partial
from typing import Any, Dict, Optional, Union
import flax
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from ..models.auto import (
from ..utils import ModelOutput, logging
from .configuration_utils import GenerationConfig
from .flax_logits_process import (
def beam_search_body_fn(state, input_ids_length=1):
    """beam search state update fn."""
    input_token = flatten_beam_dim(lax.dynamic_slice(state.running_sequences, (0, 0, state.cur_len - input_ids_length), (batch_size, num_beams, input_ids_length)))
    model_outputs = model(input_token, params=params, **state.model_kwargs)
    logits = unflatten_beam_dim(model_outputs.logits[:, -1], batch_size, num_beams)
    cache = jax.tree_util.tree_map(lambda tensor: unflatten_beam_dim(tensor, batch_size, num_beams), model_outputs.past_key_values)
    logits = self._adapt_logits_for_beam_search(logits)
    log_probs = jax.nn.log_softmax(logits)
    log_probs = logits_processor(flatten_beam_dim(running_sequences), flatten_beam_dim(log_probs), state.cur_len)
    log_probs = unflatten_beam_dim(log_probs, batch_size, num_beams)
    log_probs = log_probs + jnp.expand_dims(state.running_scores, axis=2)
    vocab_size = log_probs.shape[2]
    log_probs = log_probs.reshape((batch_size, num_beams * vocab_size))
    beams_to_keep = 2 * num_beams
    topk_log_probs, topk_indices = lax.top_k(log_probs, k=beams_to_keep)
    topk_beam_indices = topk_indices // vocab_size
    topk_running_sequences = gather_beams(state.running_sequences, topk_beam_indices, batch_size, beams_to_keep)
    topk_ids = jnp.expand_dims(topk_indices % vocab_size, axis=2)
    topk_sequences = lax.dynamic_update_slice(topk_running_sequences, topk_ids, (0, 0, state.cur_len))
    did_topk_just_finished = topk_sequences[:, :, state.cur_len] == eos_token_id
    running_topk_log_probs = topk_log_probs + did_topk_just_finished * np.array(-10000000.0)
    next_topk_indices = lax.top_k(running_topk_log_probs, k=num_beams)[1]
    next_running_sequences, next_running_scores = gather_beams([topk_sequences, running_topk_log_probs], next_topk_indices, batch_size, num_beams)
    topk_log_probs = topk_log_probs / (state.cur_len + 1 - decoder_prompt_len) ** length_penalty
    beams_in_batch_are_full = jnp.broadcast_to(state.is_sent_finished.all(axis=-1, keepdims=True), did_topk_just_finished.shape) & (early_stopping is True)
    add_penalty = ~did_topk_just_finished | beams_in_batch_are_full
    topk_log_probs += add_penalty * np.array(-10000000.0)
    merged_sequences = jnp.concatenate([state.sequences, topk_sequences], axis=1)
    merged_scores = jnp.concatenate([state.scores, topk_log_probs], axis=1)
    merged_is_sent_finished = jnp.concatenate([state.is_sent_finished, did_topk_just_finished], axis=1)
    topk_merged_indices = lax.top_k(merged_scores, k=num_beams)[1]
    next_sequences, next_scores, next_is_sent_finished = gather_beams([merged_sequences, merged_scores, merged_is_sent_finished], topk_merged_indices, batch_size, num_beams)
    next_running_indices = gather_beams(topk_beam_indices, next_topk_indices, batch_size, num_beams)
    next_cache = gather_beams(cache, next_running_indices, batch_size, num_beams)
    model_outputs['past_key_values'] = jax.tree_util.tree_map(lambda x: flatten_beam_dim(x), next_cache)
    next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)
    return BeamSearchState(cur_len=state.cur_len + 1, running_scores=next_running_scores, running_sequences=next_running_sequences, scores=next_scores, sequences=next_sequences, is_sent_finished=next_is_sent_finished, model_kwargs=next_model_kwargs)