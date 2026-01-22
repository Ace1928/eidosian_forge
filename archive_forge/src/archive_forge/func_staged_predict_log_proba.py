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
def staged_predict_log_proba(self, data, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):
    """
        Predict classification target at each stage for data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        eval_period: int, optional (default=1)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : generator for each iteration that generates:
            If data is for a single object
                return one-dimensional numpy.ndarray with log probability for every class.
            otherwise
                return two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                with log probability for every class for each object.
        """
    return self._staged_predict(data, 'LogProbability', ntree_start, ntree_end, eval_period, thread_count, verbose, 'staged_predict_log_proba')