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
def save_borders(self, fname):
    """
        Save the model borders to a file.

        Parameters
        ----------
        fname : string or pathlib.Path
            Output file name.
        """
    if not isinstance(fname, PATH_TYPES):
        raise CatBoostError('Invalid fname type={}: must be str() or pathlib.Path().'.format(type(fname)))
    self._save_borders(fname)