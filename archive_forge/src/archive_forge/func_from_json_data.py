import glob
import json
import logging
import math
import numpy as np
import os
from pathlib import Path
import random
import re
import tree  # pip install dm_tree
from typing import List, Optional, TYPE_CHECKING, Union
from urllib.parse import urlparse
import zipfile
from ray.rllib.offline.input_reader import InputReader
from ray.rllib.offline.io_context import IOContext
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import (
from ray.rllib.utils.annotations import override, PublicAPI, DeveloperAPI
from ray.rllib.utils.compression import unpack_if_needed
from ray.rllib.utils.spaces.space_utils import clip_action, normalize_action
from ray.rllib.utils.typing import Any, FileType, SampleBatchType
@DeveloperAPI
def from_json_data(json_data: Any, worker: Optional['RolloutWorker']):
    if 'type' in json_data:
        data_type = json_data.pop('type')
    else:
        raise ValueError("JSON record missing 'type' field")
    if data_type == 'SampleBatch':
        if worker is not None and len(worker.policy_map) != 1:
            raise ValueError('Found single-agent SampleBatch in input file, but our PolicyMap contains more than 1 policy!')
        for k, v in json_data.items():
            json_data[k] = unpack_if_needed(v)
        if worker is not None:
            policy = next(iter(worker.policy_map.values()))
            json_data = _adjust_obs_actions_for_policy(json_data, policy)
        json_data = _adjust_dones(json_data)
        return SampleBatch(json_data)
    elif data_type == 'MultiAgentBatch':
        policy_batches = {}
        for policy_id, policy_batch in json_data['policy_batches'].items():
            inner = {}
            for k, v in policy_batch.items():
                if k == SampleBatch.DONES:
                    k = SampleBatch.TERMINATEDS
                inner[k] = unpack_if_needed(v)
            if worker is not None:
                policy = worker.policy_map[policy_id]
                inner = _adjust_obs_actions_for_policy(inner, policy)
            inner = _adjust_dones(inner)
            policy_batches[policy_id] = SampleBatch(inner)
        return MultiAgentBatch(policy_batches, json_data['count'])
    else:
        raise ValueError("Type field must be one of ['SampleBatch', 'MultiAgentBatch']", data_type)