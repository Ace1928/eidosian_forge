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
def file_or_filename(input):
    """Open a filename for reading with `smart_open`, or seek to the beginning if `input` is an already open file.

    Parameters
    ----------
    input : str or file-like
        Filename or file-like object.

    Returns
    -------
    file-like object
        An open file, positioned at the beginning.

    """
    if isinstance(input, str):
        return open(input, 'rb')
    else:
        input.seek(0)
        return input