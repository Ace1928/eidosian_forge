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
@PublicAPI
def concat_samples_into_ma_batch(samples: List[SampleBatchType]) -> 'MultiAgentBatch':
    """Concatenates a list of SampleBatchTypes to a single MultiAgentBatch type.

    This function, as opposed to concat_samples() forces the output to always be
    MultiAgentBatch which is more generic than SampleBatch.

    Args:
        samples: List of SampleBatches or MultiAgentBatches to be
            concatenated.

    Returns:
        A new (concatenated) MultiAgentBatch.

    .. testcode::
        :skipif: True

        import numpy as np
        from ray.rllib.policy.sample_batch import SampleBatch
        b1 = MultiAgentBatch({'default_policy': {
                                        "a": np.array([1, 2]),
                                        "b": np.array([10, 11])
                                        }}, env_steps=2)
        b2 = SampleBatch({"a": np.array([3]),
                          "b": np.array([12])})
        print(concat_samples([b1, b2]))

    .. testoutput::

        {'default_policy': {"a": np.array([1, 2, 3]),
                            "b": np.array([10, 11, 12])}}

    """
    policy_batches = collections.defaultdict(list)
    env_steps = 0
    for s in samples:
        if isinstance(s, SampleBatch):
            if len(s) <= 0:
                continue
            else:
                s = s.as_multi_agent()
        elif not isinstance(s, MultiAgentBatch):
            raise ValueError('`concat_samples_into_ma_batch` can only concat SampleBatch|MultiAgentBatch objects, not {}!'.format(type(s).__name__))
        for key, batch in s.policy_batches.items():
            policy_batches[key].append(batch)
        env_steps += s.env_steps()
    out = {}
    for key, batches in policy_batches.items():
        out[key] = concat_samples(batches)
    return MultiAgentBatch(out, env_steps)