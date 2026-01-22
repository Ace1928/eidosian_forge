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
def set_subgroup_id(self, subgroup_id):
    self._check_subgroup_id_type(subgroup_id)
    subgroup_id = self._if_pandas_to_numpy(subgroup_id)
    self._check_subgroup_id_shape(subgroup_id, self.num_row())
    self._set_subgroup_id(subgroup_id)
    return self