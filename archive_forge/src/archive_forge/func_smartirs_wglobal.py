import logging
from functools import partial
import re
import numpy as np
from gensim import interfaces, matutils, utils
from gensim.utils import deprecated
def smartirs_wglobal(docfreq, totaldocs, global_scheme):
    """Calculate global document weight based on the weighting scheme specified in `global_scheme`.

    Parameters
    ----------
    docfreq : int
        Document frequency.
    totaldocs : int
        Total number of documents.
    global_scheme : {'n', 'f', 't', 'p'}
        Global transformation scheme.

    Returns
    -------
    float
        Calculated global weight.

    """
    if global_scheme == 'n':
        return 1.0
    elif global_scheme == 'f':
        return np.log2(1.0 * totaldocs / docfreq)
    elif global_scheme == 't':
        return np.log2((totaldocs + 1.0) / docfreq)
    elif global_scheme == 'p':
        return max(0, np.log2((1.0 * totaldocs - docfreq) / docfreq))