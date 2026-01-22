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
def mock_data(n_items=1000, dim=1000, prob_nnz=0.5, lam=1.0):
    """Create a random Gensim-style corpus (BoW), using :func:`~gensim.utils.mock_data_row`.

    Parameters
    ----------
    n_items : int
        Size of corpus
    dim : int
        Dimension of vector, used for :func:`~gensim.utils.mock_data_row`.
    prob_nnz : float, optional
        Probability of each coordinate will be nonzero, will be drawn from Poisson distribution,
        used for :func:`~gensim.utils.mock_data_row`.
    lam : float, optional
        Parameter for Poisson distribution, used for :func:`~gensim.utils.mock_data_row`.

    Returns
    -------
    list of list of (int, float)
        Gensim-style corpus.

    """
    return [mock_data_row(dim=dim, prob_nnz=prob_nnz, lam=lam) for _ in range(n_items)]