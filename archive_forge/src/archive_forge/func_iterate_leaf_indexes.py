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
def iterate_leaf_indexes(self, data, ntree_start=0, ntree_end=0):
    """
        Returns indexes of leafs to which objects from pool are mapped by model trees.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        ntree_start: int, optional (default=0)
            Index of first tree for which leaf indexes will be calculated (zero-based indexing).

        ntree_end: int, optional (default=0)
            Index of the tree after last tree for which leaf indexes will be calculated (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        Returns
        -------
        leaf_indexes : generator. For each object in pool yields one-dimensional numpy.ndarray of leaf indexes.
        """
    return self._iterate_leaf_indexes(data, ntree_start, ntree_end)