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
def is_corpus(obj):
    """Check whether `obj` is a corpus, by peeking at its first element. Works even on streamed generators.
    The peeked element is put back into a object returned by this function, so always use
    that returned object instead of the original `obj`.

    Parameters
    ----------
    obj : object
        An `iterable of iterable` that contains (int, numeric).

    Returns
    -------
    (bool, object)
        Pair of (is `obj` a corpus, `obj` with peeked element restored)

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.utils import is_corpus
        >>> corpus = [[(1, 1.0)], [(2, -0.3), (3, 0.12)]]
        >>> corpus_or_not, corpus = is_corpus(corpus)

    Warnings
    --------
    An "empty" corpus (empty input sequence) is ambiguous, so in this case
    the result is forcefully defined as (False, `obj`).

    """
    try:
        if 'Corpus' in obj.__class__.__name__:
            return (True, obj)
    except Exception:
        pass
    try:
        if hasattr(obj, 'next') or hasattr(obj, '__next__'):
            doc1 = next(obj)
            obj = itertools.chain([doc1], obj)
        else:
            doc1 = next(iter(obj))
        if len(doc1) == 0:
            return (True, obj)
        id1, val1 = next(iter(doc1))
        id1, val1 = (int(id1), float(val1))
    except Exception:
        return (False, obj)
    return (True, obj)