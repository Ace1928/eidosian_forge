import collections
from functools import partial
import itertools
import sys
from numbers import Number
from typing import Dict, Iterator, Set, Union
from typing import List, Optional
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.utils.annotations import DeveloperAPI, ExperimentalAPI, PublicAPI
from ray.rllib.utils.compression import pack, unpack, is_compressed
from ray.rllib.utils.deprecation import Deprecated, deprecation_warning
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import (
from ray.util import log_once
def right_zero_pad(self, max_seq_len: int, exclude_states: bool=True):
    """Right (adding zeros at end) zero-pads this SampleBatch in-place.

        This will set the `self.zero_padded` flag to True and
        `self.max_seq_len` to the given `max_seq_len` value.

        Args:
            max_seq_len: The max (total) length to zero pad to.
            exclude_states: If False, also right-zero-pad all
                `state_in_x` data. If True, leave `state_in_x` keys
                as-is.

        Returns:
            This very (now right-zero-padded) SampleBatch.

        Raises:
            ValueError: If self[SampleBatch.SEQ_LENS] is None (not defined).

        .. testcode::
            :skipif: True

            from ray.rllib.policy.sample_batch import SampleBatch
            batch = SampleBatch(
                {"a": [1, 2, 3], "seq_lens": [1, 2]})
            print(batch.right_zero_pad(max_seq_len=4))

            batch = SampleBatch({"a": [1, 2, 3],
                                 "state_in_0": [1.0, 3.0],
                                 "seq_lens": [1, 2]})
            print(batch.right_zero_pad(max_seq_len=5))

        .. testoutput::

            {"a": [1, 0, 0, 0, 2, 3, 0, 0], "seq_lens": [1, 2]}
            {"a": [1, 0, 0, 0, 0, 2, 3, 0, 0, 0],
             "state_in_0": [1.0, 3.0],  # <- all state-ins remain as-is
             "seq_lens": [1, 2]}

        """
    seq_lens = self.get(SampleBatch.SEQ_LENS)
    if seq_lens is None:
        raise ValueError(f'Cannot right-zero-pad SampleBatch if no `seq_lens` field present! SampleBatch={self}')
    length = len(seq_lens) * max_seq_len

    def _zero_pad_in_place(path, value):
        if exclude_states is True and path[0].startswith('state_in_') or path[0] == SampleBatch.SEQ_LENS:
            return
        if value.dtype == object or value.dtype.type is np.str_:
            f_pad = [None] * length
        else:
            f_pad = np.zeros((length,) + np.shape(value)[1:], dtype=value.dtype)
        f_pad_base = f_base = 0
        for len_ in self[SampleBatch.SEQ_LENS]:
            f_pad[f_pad_base:f_pad_base + len_] = value[f_base:f_base + len_]
            f_pad_base += max_seq_len
            f_base += len_
        assert f_base == len(value), value
        curr = self
        for i, p in enumerate(path):
            if i == len(path) - 1:
                curr[p] = f_pad
            curr = curr[p]
    self_as_dict = {k: v for k, v in self.items()}
    tree.map_structure_with_path(_zero_pad_in_place, self_as_dict)
    self.zero_padded = True
    self.max_seq_len = max_seq_len
    return self