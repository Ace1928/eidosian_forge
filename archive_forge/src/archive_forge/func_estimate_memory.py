import logging
import os
from collections import namedtuple, defaultdict
from collections.abc import Iterable
from timeit import default_timer
from dataclasses import dataclass
from numpy import zeros, float32 as REAL, vstack, integer, dtype
import numpy as np
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.utils import deprecated
from gensim.models import Word2Vec, FAST_VERSION  # noqa: F401
from gensim.models.keyedvectors import KeyedVectors, pseudorandom_weak_vector
def estimate_memory(self, vocab_size=None, report=None):
    """Estimate required memory for a model using current settings.

        Parameters
        ----------
        vocab_size : int, optional
            Number of raw words in the vocabulary.
        report : dict of (str, int), optional
            A dictionary from string representations of the **specific** model's memory consuming members
            to their size in bytes.

        Returns
        -------
        dict of (str, int), optional
            A dictionary from string representations of the model's memory consuming members to their size in bytes.
            Includes members from the base classes as well as weights and tag lookup memory estimation specific to the
            class.

        """
    report = report or {}
    report['doctag_lookup'] = self.estimated_lookup_memory()
    report['doctag_syn0'] = len(self.dv) * self.vector_size * dtype(REAL).itemsize
    return super(Doc2Vec, self).estimate_memory(vocab_size, report=report)