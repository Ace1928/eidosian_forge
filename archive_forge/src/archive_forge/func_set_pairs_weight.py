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
def set_pairs_weight(self, pairs_weight):
    self._check_weight_type(pairs_weight)
    pairs_weight = self._if_pandas_to_numpy(pairs_weight)
    self._check_weight_shape(pairs_weight, self.num_pairs())
    self._set_pairs_weight(pairs_weight)
    return self