import logging
from functools import partial
import re
import numpy as np
from gensim import interfaces, matutils, utils
from gensim.utils import deprecated
def smartirs_wlocal(tf, local_scheme):
    """Calculate local term weight for a term using the weighting scheme specified in `local_scheme`.

    Parameters
    ----------
    tf : int
        Term frequency.
    local : {'b', 'n', 'a', 'l', 'd', 'L'}
        Local transformation scheme.

    Returns
    -------
    float
        Calculated local weight.

    """
    if local_scheme == 'n':
        return tf
    elif local_scheme == 'l':
        return 1 + np.log2(tf)
    elif local_scheme == 'd':
        return 1 + np.log2(1 + np.log2(tf))
    elif local_scheme == 'a':
        return 0.5 + 0.5 * tf / tf.max(axis=0)
    elif local_scheme == 'b':
        return tf.astype('bool').astype('int')
    elif local_scheme == 'L':
        return (1 + np.log2(tf)) / (1 + np.log2(tf.mean(axis=0)))