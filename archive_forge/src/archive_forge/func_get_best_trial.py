import copy
import fnmatch
import io
import json
import logging
from numbers import Number
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pyarrow.fs
from ray.util.annotations import PublicAPI
from ray.air.constants import (
from ray.train import Checkpoint
from ray.train._internal.storage import (
from ray.tune.execution.tune_controller import TuneController
from ray.tune.experiment import Trial
from ray.tune.result import (
from ray.tune.utils import flatten_dict
from ray.tune.utils.serialization import TuneFunctionDecoder
from ray.tune.utils.util import is_nan_or_inf, is_nan, unflattened_lookup
def get_best_trial(self, metric: Optional[str]=None, mode: Optional[str]=None, scope: str='last', filter_nan_and_inf: bool=True) -> Optional[Trial]:
    """Retrieve the best trial object.

        Compares all trials' scores on ``metric``.
        If ``metric`` is not specified, ``self.default_metric`` will be used.
        If `mode` is not specified, ``self.default_mode`` will be used.
        These values are usually initialized by passing the ``metric`` and
        ``mode`` parameters to ``tune.run()``.

        Args:
            metric: Key for trial info to order on. Defaults to
                ``self.default_metric``.
            mode: One of [min, max]. Defaults to ``self.default_mode``.
            scope: One of [all, last, avg, last-5-avg, last-10-avg].
                If `scope=last`, only look at each trial's final step for
                `metric`, and compare across trials based on `mode=[min,max]`.
                If `scope=avg`, consider the simple average over all steps
                for `metric` and compare across trials based on
                `mode=[min,max]`. If `scope=last-5-avg` or `scope=last-10-avg`,
                consider the simple average over the last 5 or 10 steps for
                `metric` and compare across trials based on `mode=[min,max]`.
                If `scope=all`, find each trial's min/max score for `metric`
                based on `mode`, and compare trials based on `mode=[min,max]`.
            filter_nan_and_inf: If True (default), NaN or infinite
                values are disregarded and these trials are never selected as
                the best trial.

        Returns:
            The best trial for the provided metric. If no trials contain the provided
                metric, or if the value for the metric is NaN for all trials,
                then returns None.
        """
    if len(self.trials) == 1:
        return self.trials[0]
    metric = self._validate_metric(metric)
    mode = self._validate_mode(mode)
    if scope not in ['all', 'last', 'avg', 'last-5-avg', 'last-10-avg']:
        raise ValueError('ExperimentAnalysis: attempting to get best trial for metric {} for scope {} not in ["all", "last", "avg", "last-5-avg", "last-10-avg"]. If you didn\'t pass a `metric` parameter to `tune.run()`, you have to pass one when fetching the best trial.'.format(metric, scope))
    best_trial = None
    best_metric_score = None
    for trial in self.trials:
        if metric not in trial.metric_analysis:
            continue
        if scope in ['last', 'avg', 'last-5-avg', 'last-10-avg']:
            metric_score = trial.metric_analysis[metric][scope]
        else:
            metric_score = trial.metric_analysis[metric][mode]
        if filter_nan_and_inf and is_nan_or_inf(metric_score):
            continue
        if best_metric_score is None:
            best_metric_score = metric_score
            best_trial = trial
            continue
        if mode == 'max' and best_metric_score < metric_score:
            best_metric_score = metric_score
            best_trial = trial
        elif mode == 'min' and best_metric_score > metric_score:
            best_metric_score = metric_score
            best_trial = trial
    if not best_trial:
        logger.warning('Could not find best trial. Did you pass the correct `metric` parameter?')
    return best_trial