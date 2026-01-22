from __future__ import print_function
import logging
import os
import math
import time
import numpy
import scipy.sparse as sparse
import gensim
from gensim.corpora import IndexedCorpus
from gensim.interfaces import TransformedCorpus
def get_by_offset(self, offset):
    """As opposed to getitem, this one only accepts ints as offsets."""
    self._ensure_shard(offset)
    result = self.current_shard[offset - self.current_offset]
    return result