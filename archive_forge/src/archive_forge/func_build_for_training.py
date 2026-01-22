import copy
import logging
import math
from typing import Any, Dict, List, Optional
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Space
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.spaces.space_utils import (
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
def build_for_training(self, view_requirements: ViewRequirementsDict) -> SampleBatch:
    """Builds a SampleBatch from the thus-far collected agent data.

        If the episode/trajectory has no TERMINATED|TRUNCATED=True at the end, will
        copy the necessary n timesteps at the end of the trajectory back to the
        beginning of the buffers and wait for new samples coming in.
        SampleBatches created by this method will be ready for postprocessing
        by a Policy.

        Args:
            view_requirements: The viewrequirements dict needed to build the
            SampleBatch from the raw buffers (which may have data shifts as well as
            mappings from view-col to data-col in them).

        Returns:
            SampleBatch: The built SampleBatch for this agent, ready to go into
            postprocessing.
        """
    batch_data = {}
    np_data = {}
    for view_col, view_req in view_requirements.items():
        data_col = view_req.data_col or view_col
        if data_col not in self.buffers:
            is_state = self._fill_buffer_with_initial_values(data_col, view_req, build_for_inference=False)
            if not is_state:
                continue
        obs_shift = -1 if data_col in [SampleBatch.OBS, SampleBatch.INFOS] else 0
        self._cache_in_np(np_data, data_col)
        data = []
        for d in np_data[data_col]:
            shifted_data = []
            count = int(math.ceil((len(d) - int(data_col in self.data_cols_with_dummy_values) - self.shift_before) / view_req.batch_repeat_value))
            for i in range(count):
                inds = self.shift_before + obs_shift + view_req.shift_arr + i * view_req.batch_repeat_value
                if max(inds) < len(d):
                    element_at_t = d[inds]
                else:
                    element_at_t = _get_buffered_slice_with_paddings(d, inds)
                    element_at_t = np.stack(element_at_t)
                if element_at_t.shape[0] == 1:
                    element_at_t = element_at_t[0]
                shifted_data.append(element_at_t)
            if shifted_data:
                shifted_data_np = np.stack(shifted_data, 0)
            else:
                shifted_data_np = np.array(shifted_data)
            data.append(shifted_data_np)
        batch_data[view_col] = self._unflatten_as_buffer_struct(data, data_col)
    batch = self._get_sample_batch(batch_data)
    if SampleBatch.TERMINATEDS in self.buffers and (not self.buffers[SampleBatch.TERMINATEDS][0][-1]) and (SampleBatch.TRUNCATEDS in self.buffers) and (not self.buffers[SampleBatch.TRUNCATEDS][0][-1]):
        if self.shift_before > 0:
            for k, data in self.buffers.items():
                for i in range(len(data)):
                    self.buffers[k][i] = data[i][-self.shift_before:]
        self.agent_steps = 0
    self.unroll_id = None
    return batch