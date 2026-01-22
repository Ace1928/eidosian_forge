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
@DeveloperAPI
def convert_ma_batch_to_sample_batch(batch: SampleBatchType) -> SampleBatch:
    """Converts a MultiAgentBatch to a SampleBatch if neccessary.

    Args:
        batch: The SampleBatchType to convert.

    Returns:
        batch: the converted SampleBatch

    Raises:
        ValueError if the MultiAgentBatch has more than one policy_id
        or if the policy_id is not `DEFAULT_POLICY_ID`
    """
    if isinstance(batch, MultiAgentBatch):
        policy_keys = batch.policy_batches.keys()
        if len(policy_keys) == 1 and DEFAULT_POLICY_ID in policy_keys:
            batch = batch.policy_batches[DEFAULT_POLICY_ID]
        else:
            raise ValueError('RLlib tried to convert a multi agent-batch with data from more than one policy to a single-agent batch. This is not supported and may be due to a number of issues. Here are two possible ones:1) Off-Policy Estimation is not implemented for multi-agent batches. You can set `off_policy_estimation_methods: {}` to resolve this.2) Loading multi-agent data for offline training is not implemented.Load single-agent data instead to resolve this.')
    return batch