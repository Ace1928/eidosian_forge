import logging
import os
from typing import Any, Callable, Dict, Optional, Type, Union, TYPE_CHECKING
import pyarrow.fs
import ray
from ray.air.config import RunConfig
from ray.air._internal.usage import AirEntrypoint
from ray.air.util.node import _force_on_current_node
from ray.train._internal.storage import _exists_at_fs_path, get_fs_and_path
from ray.tune import TuneError
from ray.tune.execution.experiment_state import _ResumeConfig
from ray.tune.experimental.output import (
from ray.tune.result_grid import ResultGrid
from ray.tune.trainable import Trainable
from ray.tune.impl.tuner_internal import TunerInternal, _TUNER_PKL
from ray.tune.tune_config import TuneConfig
from ray.tune.progress_reporter import (
from ray.util import PublicAPI
Get results of a hyperparameter tuning run.

        This method returns the same results as :meth:`fit() <ray.tune.tuner.Tuner.fit>`
        and can be used to retrieve the results after restoring a tuner without
        calling ``fit()`` again.

        If the tuner has not been fit before, an error will be raised.

        .. code-block:: python

            from ray.tune import Tuner

            # `trainable` is what was passed in to the original `Tuner`
            tuner = Tuner.restore("/path/to/experiment', trainable=trainable)
            results = tuner.get_results()

        Returns:
            Result grid of a previously fitted tuning run.

        