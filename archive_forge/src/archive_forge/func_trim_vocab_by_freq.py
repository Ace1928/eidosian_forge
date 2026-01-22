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
def trim_vocab_by_freq(vocab, topk, trim_rule=None):
    """Retain `topk` most frequent words in `vocab`.
    If there are more words with the same frequency as `topk`-th one, they will be kept.
    Modifies `vocab` in place, returns nothing.

    Parameters
    ----------
    vocab : dict
        Input dictionary.
    topk : int
        Number of words with highest frequencies to keep.
    trim_rule : function, optional
        Function for trimming entities from vocab, default behaviour is `vocab[w] <= min_count`.

    """
    if topk >= len(vocab):
        return
    min_count = heapq.nlargest(topk, vocab.values())[-1]
    prune_vocab(vocab, min_count, trim_rule=trim_rule)