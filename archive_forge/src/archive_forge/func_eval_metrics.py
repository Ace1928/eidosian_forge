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
def eval_metrics(self, data, metrics, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, tmp_dir=None, plot=False, plot_file=None, log_cout=None, log_cerr=None):
    """
        Calculate metrics.

        Parameters
        ----------
        data : catboost.Pool
            Data to evaluate metrics on.

        metrics : list of strings or catboost.metrics.BuiltinMetric
            List of evaluated metrics.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        eval_period: int, optional (default=1)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        tmp_dir : string or pathlib.Path (default=None)
            The name of the temporary directory for intermediate results.
            If None, then the name will be generated.

        plot : bool, optional (default=False)
            If True, draw train and eval error in Jupyter notebook

        plot_file : file-like or str, optional (default=None)
            If not None, save train and eval error graphs to file

        log_cout: output stream or callback for logging (default=None)
            If None is specified, sys.stdout is used

        log_cerr: error stream or callback for logging (default=None)
            If None is specified, sys.stderr is used

        Returns
        -------
        prediction : dict: metric -> array of shape [(ntree_end - ntree_start) / eval_period]
        """
    return self._eval_metrics(data, metrics, ntree_start, ntree_end, eval_period, thread_count, _get_train_dir(self.get_params()), tmp_dir, plot, plot_file, log_cout, log_cerr)