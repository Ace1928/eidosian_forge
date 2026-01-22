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
def setup_rmm_pool(_: Any, pytestconfig: pytest.Config) -> None:
    if pytestconfig.getoption('--use-rmm-pool'):
        if no_rmm()['condition']:
            raise ImportError('The --use-rmm-pool option requires the RMM package')
        if no_dask_cuda()['condition']:
            raise ImportError('The --use-rmm-pool option requires the dask_cuda package')
        import rmm
        from dask_cuda.utils import get_n_gpus
        rmm.reinitialize(pool_allocator=True, initial_pool_size=1024 * 1024 * 1024, devices=list(range(get_n_gpus())))