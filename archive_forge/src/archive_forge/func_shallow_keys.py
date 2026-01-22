from collections import abc
import itertools
from typing import (
from ray.rllib.utils.annotations import ExperimentalAPI
def shallow_keys(self) -> AbstractSet[str]:
    """Returns a set of the keys at the top level of the NestedDict."""
    return self._data.keys()