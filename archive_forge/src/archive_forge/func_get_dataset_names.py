import os
import inspect
import warnings
import colorsys
from contextlib import contextmanager
from urllib.request import urlopen, urlretrieve
from types import ModuleType
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
from matplotlib.cbook import normalize_kwargs
from seaborn._core.typing import deprecated
from seaborn.external.version import Version
from seaborn.external.appdirs import user_cache_dir
def get_dataset_names():
    """Report available example datasets, useful for reporting issues.

    Requires an internet connection.

    """
    with urlopen(DATASET_NAMES_URL) as resp:
        txt = resp.read()
    dataset_names = [name.strip() for name in txt.decode().split('\n')]
    return list(filter(None, dataset_names))