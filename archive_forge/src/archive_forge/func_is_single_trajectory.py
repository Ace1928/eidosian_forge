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
@ExperimentalAPI
def is_single_trajectory(self) -> bool:
    """Returns True if this SampleBatch only contains one trajectory.

        This is determined by checking all timesteps (except for the last) for being
        not terminated AND (if applicable) not truncated.
        """
    return not any(self[SampleBatch.TERMINATEDS][:-1]) and (SampleBatch.TRUNCATEDS not in self or not any(self[SampleBatch.TRUNCATEDS][:-1]))