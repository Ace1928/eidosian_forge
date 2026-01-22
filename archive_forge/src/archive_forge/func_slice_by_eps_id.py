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
def slice_by_eps_id():
    slices = []
    cur_eps_id = self[SampleBatch.EPS_ID][0]
    offset = 0
    for i in range(self.count):
        next_eps_id = self[SampleBatch.EPS_ID][i]
        if next_eps_id != cur_eps_id:
            slices.append(self[offset:i])
            offset = i
            cur_eps_id = next_eps_id
    slices.append(self[offset:self.count])
    return slices