import abc
import json
from copy import deepcopy
from inspect import signature
from typing import Dict, List, Union
from dataclasses import dataclass
import ray
from ray.util import placement_group
from ray.util.annotations import DeveloperAPI
def to_placement_group(self):
    return placement_group(*self._bound.args, **self._bound.kwargs)