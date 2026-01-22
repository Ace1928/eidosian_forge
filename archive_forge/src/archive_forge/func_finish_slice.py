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
def finish_slice():
    nonlocal cur_slice_size
    assert cur_slice_size > 0
    batch = MultiAgentBatch({k: v.build_and_reset() for k, v in cur_slice.items()}, cur_slice_size)
    cur_slice_size = 0
    cur_slice.clear()
    finished_slices.append(batch)