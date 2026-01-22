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
def simple_preprocess(doc, deacc=False, min_len=2, max_len=15):
    """Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long.

    Uses :func:`~gensim.utils.tokenize` internally.

    Parameters
    ----------
    doc : str
        Input document.
    deacc : bool, optional
        Remove accent marks from tokens using :func:`~gensim.utils.deaccent`?
    min_len : int, optional
        Minimum length of token (inclusive). Shorter tokens are discarded.
    max_len : int, optional
        Maximum length of token in result (inclusive). Longer tokens are discarded.

    Returns
    -------
    list of str
        Tokens extracted from `doc`.

    """
    tokens = [token for token in tokenize(doc, lower=True, deacc=deacc, errors='ignore') if min_len <= len(token) <= max_len and (not token.startswith('_'))]
    return tokens