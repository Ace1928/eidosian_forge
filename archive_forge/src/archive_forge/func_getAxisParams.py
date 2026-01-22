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
def getAxisParams(borders, feature_name=None):
    return {'title': 'Bins' if feature_name is None else "Bins of feature '{}'".format(feature_name), 'tickmode': 'array', 'tickvals': list(range(len(borders) + 1)), 'ticktext': ['(-inf, {:.4f}]'.format(borders[0])] + ['({:.4f}, {:.4f}]'.format(val_1, val_2) for val_1, val_2 in zip(borders[:-1], borders[1:])] + ['({:.4f}, +inf)'.format(borders[-1])], 'showticklabels': False}