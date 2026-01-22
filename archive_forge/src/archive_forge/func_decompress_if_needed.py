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
def decompress_if_needed(self, columns: Set[str]=frozenset(['obs', 'new_obs'])) -> 'MultiAgentBatch':
    """Decompresses each policy batch (per column), if already compressed.

        Args:
            columns: Set of column names to decompress.

        Returns:
            Self.
        """
    for batch in self.policy_batches.values():
        batch.decompress_if_needed(columns)
    return self