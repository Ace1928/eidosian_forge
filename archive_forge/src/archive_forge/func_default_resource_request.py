import copy
from datetime import datetime
import logging
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import ray
import ray.cloudpickle as ray_pickle
from ray.air._internal.util import skip_exceptions, exception_cause
from ray.air.constants import (
from ray.train._internal.checkpoint_manager import _TrainingResult
from ray.train._internal.storage import StorageContext, _exists_at_fs_path
from ray.train import Checkpoint
from ray.tune.result import (
from ray.tune.utils import UtilMonitor
from ray.tune.utils.log import disable_ipython
from ray.tune.utils.util import Tee
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.util.annotations import DeveloperAPI, PublicAPI
@classmethod
def default_resource_request(cls, config: Dict[str, Any]) -> Optional[PlacementGroupFactory]:
    """Provides a static resource requirement for the given configuration.

        This can be overridden by sub-classes to set the correct trial resource
        allocation, so the user does not need to.

        .. testcode::

            @classmethod
            def default_resource_request(cls, config):
                return PlacementGroupFactory([{"CPU": 1}, {"CPU": 1}])


        Args:
            config[Dict[str, Any]]: The Trainable's config dict.

        Returns:
            PlacementGroupFactory: A PlacementGroupFactory consumed by Tune
                for queueing.
        """
    return None