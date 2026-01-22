from contextlib import contextmanager  # noqa E402
from copy import deepcopy
import logging
import sys
import os
from collections import OrderedDict, defaultdict
from six import iteritems, string_types, integer_types
import warnings
import numpy as np
import ctypes
import platform
import tempfile
import shutil
import json
from enum import Enum
from operator import itemgetter
import threading
import scipy.sparse
from .plot_helpers import save_plot_file, try_plot_offline, OfflineMetricVisualizer
from . import _catboost
from .metrics import BuiltinMetric
def create_metric_calcer(self, metrics, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, tmp_dir=None):
    """
        Create batch metric calcer. Could be used to aggregate metric on several pools
        Parameters
        ----------
            Same as in eval_metrics except data
        Returns
        -------
            BatchMetricCalcer object

        Usage example
        -------
        # Large dataset is partitioned into parts [part1, part2]
        model.fit(params)
        batch_calcer = model.create_metric_calcer(['Logloss'])
        batch_calcer.add(part1)
        batch_calcer.add(part2)
        metrics = batch_calcer.eval_metrics()
        """
    if not self.is_fitted():
        raise CatBoostError('There is no trained model to evaluate metrics on. Use fit() to train model. Then call this method.')
    return BatchMetricCalcer(self._object, metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir)