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
def stringify_builtin_metrics(params):
    """Replace all occurrences of BuiltinMetric with their string representations."""
    for f in ['loss_function', 'objective', 'eval_metric', 'custom_metric', 'custom_loss']:
        if f not in params:
            continue
        val = params[f]
        if isinstance(val, BuiltinMetric):
            params[f] = str(val)
        elif isinstance(val, STRING_TYPES):
            continue
        elif isinstance(val, Sequence):
            params[f] = stringify_builtin_metrics_list(val)
    return params