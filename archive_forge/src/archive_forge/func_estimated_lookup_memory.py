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
def estimated_lookup_memory(self):
    """Get estimated memory for tag lookup, 0 if using pure int tags.

        Returns
        -------
        int
            The estimated RAM required to look up a tag in bytes.

        """
    return 60 * len(self.dv) + 140 * len(self.dv)