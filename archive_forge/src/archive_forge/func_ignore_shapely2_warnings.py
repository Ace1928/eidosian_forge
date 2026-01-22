import contextlib
from packaging.version import Version
import importlib
import os
import warnings
import numpy as np
import pandas as pd
import shapely
import shapely.geos
@contextlib.contextmanager
def ignore_shapely2_warnings():
    yield