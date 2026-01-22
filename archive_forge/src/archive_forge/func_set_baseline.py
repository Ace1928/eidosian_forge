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
def set_baseline(self, baseline):
    self._check_baseline_type(baseline)
    baseline = self._if_pandas_to_numpy(baseline)
    baseline = np.reshape(baseline, (self.num_row(), -1))
    self._check_baseline_shape(baseline, self.num_row())
    self._set_baseline(baseline)
    return self