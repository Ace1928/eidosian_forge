from collections import namedtuple
import logging
import json
from typing import Optional
from ray.tune.execution.placement_groups import (
from ray.tune.utils.resource_updater import _Resources
from ray.util.annotations import Deprecated, DeveloperAPI
from ray.tune import TuneError
@Deprecated
def resources_to_json(*args, **kwargs):
    raise DeprecationWarning('tune.Resources is depracted. Use tune.PlacementGroupFactory instead.')