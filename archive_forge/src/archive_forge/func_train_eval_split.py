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
def train_eval_split(self, has_time, is_classification, eval_fraction, save_eval_pool):
    train_pool = Pool(None, data_can_be_none=True)
    eval_pool = Pool(None, data_can_be_none=True)
    self._train_eval_split(train_pool, eval_pool, has_time, is_classification, eval_fraction, save_eval_pool)
    return (train_pool, eval_pool)