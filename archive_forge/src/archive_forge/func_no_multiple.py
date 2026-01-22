import gc
import importlib.util
import multiprocessing
import os
import platform
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import StringIO
from platform import system
from typing import (
import numpy as np
import pytest
from scipy import sparse
import xgboost as xgb
from xgboost.core import ArrayLike
from xgboost.sklearn import SklObjective
from xgboost.testing.data import (
from hypothesis import strategies
from hypothesis.extra.numpy import arrays
def no_multiple(*args: Any) -> PytestSkip:
    condition = False
    reason = ''
    for arg in args:
        condition = condition or arg['condition']
        if arg['condition']:
            reason = arg['reason']
            break
    return {'condition': condition, 'reason': reason}