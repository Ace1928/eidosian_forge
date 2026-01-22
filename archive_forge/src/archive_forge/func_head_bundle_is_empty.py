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
def head_bundle_is_empty(self):
    """Returns True if head bundle is empty while child bundles
        need resources.

        This is considered an internal API within Tune.
        """
    return self._head_bundle_is_empty