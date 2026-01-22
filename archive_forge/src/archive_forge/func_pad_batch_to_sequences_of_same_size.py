import logging
import numpy as np
import tree  # pip install dm_tree
from typing import List, Optional
import functools
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.typing import TensorType, ViewRequirementsDict
from ray.util import log_once
from ray.rllib.utils.typing import SampleBatchType
@DeveloperAPI
def pad_batch_to_sequences_of_same_size(batch: SampleBatch, max_seq_len: int, shuffle: bool=False, batch_divisibility_req: int=1, feature_keys: Optional[List[str]]=None, view_requirements: Optional[ViewRequirementsDict]=None, _enable_new_api_stack: bool=False, padding: str='zero'):
    """Applies padding to `batch` so it's choppable into same-size sequences.

    Shuffles `batch` (if desired), makes sure divisibility requirement is met,
    then pads the batch ([B, ...]) into same-size chunks ([B, ...]) w/o
    adding a time dimension (yet).
    Padding depends on episodes found in batch and `max_seq_len`.

    Args:
        batch: The SampleBatch object. All values in here have
            the shape [B, ...].
        max_seq_len: The max. sequence length to use for chopping.
        shuffle: Whether to shuffle batch sequences. Shuffle may
            be done in-place. This only makes sense if you're further
            applying minibatch SGD after getting the outputs.
        batch_divisibility_req: The int by which the batch dimension
            must be dividable.
        feature_keys: An optional list of keys to apply sequence-chopping
            to. If None, use all keys in batch that are not
            "state_in/out_"-type keys.
        view_requirements: An optional Policy ViewRequirements dict to
            be able to infer whether e.g. dynamic max'ing should be
            applied over the seq_lens.
        _enable_new_api_stack: This is a temporary flag to enable the new RLModule API.
            After a complete rollout of the new API, this flag will be removed.
        padding: Padding type to use. Either "zero" or "last". Zero padding
            will pad with zeros, last padding will pad with the last value.
    """
    if batch.zero_padded:
        return
    batch.zero_padded = True
    if batch_divisibility_req > 1:
        meets_divisibility_reqs = len(batch[SampleBatch.CUR_OBS]) % batch_divisibility_req == 0 and max(batch[SampleBatch.AGENT_INDEX]) == 0
    else:
        meets_divisibility_reqs = True
    states_already_reduced_to_init = False
    if _enable_new_api_stack and ('state_in' in batch or 'state_out' in batch):
        seq_lens = batch.get(SampleBatch.SEQ_LENS)
        state_ins = tree.flatten(batch['state_in'])
        if state_ins:
            assert all((len(state_in) == len(state_ins[0]) for state_in in state_ins)), 'All state_in tensors should have the same batch_dim size.'
            if len(state_ins[0]) == len(seq_lens):
                states_already_reduced_to_init = True
            dynamic_max = True
        else:
            dynamic_max = False
    elif not _enable_new_api_stack and ('state_in_0' in batch or 'state_out_0' in batch):
        if batch.get(SampleBatch.SEQ_LENS) is not None and len(batch['state_in_0']) == len(batch[SampleBatch.SEQ_LENS]):
            states_already_reduced_to_init = True
        if view_requirements['state_in_0'].shift_from is None:
            dynamic_max = True
        else:
            dynamic_max = False
    elif not meets_divisibility_reqs:
        max_seq_len = batch_divisibility_req
        dynamic_max = False
        batch.max_seq_len = max_seq_len
    else:
        if shuffle:
            batch.shuffle()
        return
    state_keys = []
    feature_keys_ = feature_keys or []
    for k, v in batch.items():
        if k.startswith('state_in'):
            state_keys.append(k)
        elif not feature_keys and (not k.startswith('state_out') if not _enable_new_api_stack else True) and (k not in [SampleBatch.SEQ_LENS]):
            feature_keys_.append(k)
    feature_sequences, initial_states, seq_lens = chop_into_sequences(feature_columns=[batch[k] for k in feature_keys_], state_columns=[batch[k] for k in state_keys], episode_ids=batch.get(SampleBatch.EPS_ID), unroll_ids=batch.get(SampleBatch.UNROLL_ID), agent_indices=batch.get(SampleBatch.AGENT_INDEX), seq_lens=batch.get(SampleBatch.SEQ_LENS), max_seq_len=max_seq_len, dynamic_max=dynamic_max, states_already_reduced_to_init=states_already_reduced_to_init, shuffle=shuffle, handle_nested_data=True, padding=padding)
    for i, k in enumerate(feature_keys_):
        batch[k] = tree.unflatten_as(batch[k], feature_sequences[i])
    for i, k in enumerate(state_keys):
        batch[k] = initial_states[i]
    batch[SampleBatch.SEQ_LENS] = np.array(seq_lens)
    if dynamic_max:
        batch.max_seq_len = max(seq_lens)
    if log_once('rnn_ma_feed_dict'):
        logger.info('Padded input for RNN/Attn.Nets/MA:\n\n{}\n'.format(summarize({'features': feature_sequences, 'initial_states': initial_states, 'seq_lens': seq_lens, 'max_seq_len': max_seq_len})))