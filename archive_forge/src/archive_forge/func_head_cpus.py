import abc
import json
from copy import deepcopy
from inspect import signature
from typing import Dict, List, Union
from dataclasses import dataclass
import ray
from ray.util import placement_group
from ray.util.annotations import DeveloperAPI
@property
@DeveloperAPI
def head_cpus(self) -> float:
    """Returns the number of cpus in the head bundle."""
    return 0.0 if self._head_bundle_is_empty else self._bundles[0].get('CPU', 0.0)