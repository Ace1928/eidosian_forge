import collections.abc
import logging
import numpy as np
import scipy.sparse
from scipy.stats import halfnorm
from gensim import interfaces
from gensim import matutils
from gensim import utils
from gensim.interfaces import TransformedCorpus
from gensim.models import basemodel, CoherenceModel
from gensim.models.nmf_pgd import solve_h
An optimized version of 0.5 * trace(WtWA) - trace(WtB).