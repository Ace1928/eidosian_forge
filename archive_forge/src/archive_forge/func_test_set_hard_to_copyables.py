from __future__ import annotations
import os
import pathlib
import site
import stat
import sys
from collections import OrderedDict
from contextlib import contextmanager
import pytest
import yaml
import dask.config
from dask.config import (
def test_set_hard_to_copyables():
    import threading
    with dask.config.set(x=threading.Lock()):
        with dask.config.set(y=1):
            pass