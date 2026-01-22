import itertools
import types
import numpy as np
import pandas as pd
from .._utils import is_scalar
from ..doctools import document
from ..mapping.evaluation import after_stat
from .binning import fuzzybreaks
from .stat import stat

    Return duplicate of parameter value

    Used to apply same value to x & y axes if only one
    value is given.
    