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
def rows(self) -> Iterator[Dict[str, TensorType]]:
    """Returns an iterator over data rows, i.e. dicts with column values.

        Note that if `seq_lens` is set in self, we set it to 1 in the rows.

        Yields:
            The column values of the row in this iteration.

        .. testcode::
            :skipif: True

            from ray.rllib.policy.sample_batch import SampleBatch
            batch = SampleBatch({
               "a": [1, 2, 3],
               "b": [4, 5, 6],
               "seq_lens": [1, 2]
            })
            for row in batch.rows():
                print(row)

        .. testoutput::

            {"a": 1, "b": 4, "seq_lens": 1}
            {"a": 2, "b": 5, "seq_lens": 1}
            {"a": 3, "b": 6, "seq_lens": 1}
        """
    seq_lens = None if self.get(SampleBatch.SEQ_LENS, 1) is None else 1
    self_as_dict = {k: v for k, v in self.items()}
    for i in range(self.count):
        yield tree.map_structure_with_path(lambda p, v: v[i] if p[0] != self.SEQ_LENS else seq_lens, self_as_dict)