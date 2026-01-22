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
def get_visible_arg(show_indices, show_names):
    visible_arg = [True]
    if len(main_indices) > 0:
        visible_arg.append(True)
    if indices_present:
        visible_arg.append(show_indices)
    if names_present:
        visible_arg.append(show_names)
    if cost_graph is not None:
        visible_arg.append(True)
        visible_arg.append(show_names)
    return visible_arg