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
def timeslices(self, k: int) -> List['MultiAgentBatch']:
    """Returns k-step batches holding data for each agent at those steps.

        For examples, suppose we have agent1 observations [a1t1, a1t2, a1t3],
        for agent2, [a2t1, a2t3], and for agent3, [a3t3] only.

        Calling timeslices(1) would return three MultiAgentBatches containing
        [a1t1, a2t1], [a1t2], and [a1t3, a2t3, a3t3].

        Calling timeslices(2) would return two MultiAgentBatches containing
        [a1t1, a1t2, a2t1], and [a1t3, a2t3, a3t3].

        This method is used to implement "lockstep" replay mode. Note that this
        method does not guarantee each batch contains only data from a single
        unroll. Batches might contain data from multiple different envs.
        """
    from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
    steps = []
    for policy_id, batch in self.policy_batches.items():
        for row in batch.rows():
            steps.append((row[SampleBatch.EPS_ID], row[SampleBatch.T], row[SampleBatch.AGENT_INDEX], policy_id, row))
    steps.sort()
    finished_slices = []
    cur_slice = collections.defaultdict(SampleBatchBuilder)
    cur_slice_size = 0

    def finish_slice():
        nonlocal cur_slice_size
        assert cur_slice_size > 0
        batch = MultiAgentBatch({k: v.build_and_reset() for k, v in cur_slice.items()}, cur_slice_size)
        cur_slice_size = 0
        cur_slice.clear()
        finished_slices.append(batch)
    for _, group in itertools.groupby(steps, lambda x: x[:2]):
        for _, _, _, policy_id, row in group:
            cur_slice[policy_id].add_values(**row)
        cur_slice_size += 1
        if cur_slice_size >= k:
            finish_slice()
            assert cur_slice_size == 0
    if cur_slice_size > 0:
        finish_slice()
    assert len(finished_slices) > 0, finished_slices
    return finished_slices