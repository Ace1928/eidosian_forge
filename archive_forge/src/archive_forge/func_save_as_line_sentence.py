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
def save_as_line_sentence(corpus, filename):
    """Save the corpus in LineSentence format, i.e. each sentence on a separate line,
    tokens are separated by space.

    Parameters
    ----------
    corpus : iterable of iterables of strings

    """
    with open(filename, mode='wb', encoding='utf8') as fout:
        for sentence in corpus:
            line = any2unicode(' '.join(sentence) + '\n')
            fout.write(line)