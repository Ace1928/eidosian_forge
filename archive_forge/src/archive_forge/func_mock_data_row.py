from __future__ import with_statement
from contextlib import contextmanager
import collections.abc
import logging
import warnings
import numbers
from html.entities import name2codepoint as n2cp
import pickle as _pickle
import re
import unicodedata
import os
import random
import itertools
import tempfile
from functools import wraps
import multiprocessing
import shutil
import sys
import subprocess
import inspect
import heapq
from copy import deepcopy
from datetime import datetime
import platform
import types
import numpy as np
import scipy.sparse
from smart_open import open
from gensim import __version__ as gensim_version
def mock_data_row(dim=1000, prob_nnz=0.5, lam=1.0):
    """Create a random gensim BoW vector, with the feature counts following the Poisson distribution.

    Parameters
    ----------
    dim : int, optional
        Dimension of vector.
    prob_nnz : float, optional
        Probability of each coordinate will be nonzero, will be drawn from the Poisson distribution.
    lam : float, optional
        Lambda parameter for the Poisson distribution.

    Returns
    -------
    list of (int, float)
        Vector in BoW format.

    """
    nnz = np.random.uniform(size=(dim,))
    return [(i, float(np.random.poisson(lam=lam) + 1.0)) for i in range(dim) if nnz[i] < prob_nnz]