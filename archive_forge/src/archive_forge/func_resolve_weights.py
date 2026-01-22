import logging
from functools import partial
import re
import numpy as np
from gensim import interfaces, matutils, utils
from gensim.utils import deprecated
def resolve_weights(smartirs):
    """Check the validity of `smartirs` parameters.

    Parameters
    ----------
    smartirs : str
        `smartirs` or SMART (System for the Mechanical Analysis and Retrieval of Text)
        Information Retrieval System, a mnemonic scheme for denoting tf-idf weighting
        variants in the vector space model. The mnemonic for representing a combination
        of weights takes the form ddd, where the letters represents the term weighting of the document vector.
        for more information visit `SMART Information Retrieval System
        <https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System>`_.

    Returns
    -------
    str of (local_letter, global_letter, normalization_letter)

    local_letter : str
        Term frequency weighing, one of:
            * `b` - binary,
            * `t` or `n` - raw,
            * `a` - augmented,
            * `l` - logarithm,
            * `d` - double logarithm,
            * `L` - log average.
    global_letter : str
        Document frequency weighting, one of:
            * `x` or `n` - none,
            * `f` - idf,
            * `t` - zero-corrected idf,
            * `p` - probabilistic idf.
    normalization_letter : str
        Document normalization, one of:
            * `x` or `n` - none,
            * `c` - cosine,
            * `u` - pivoted unique,
            * `b` - pivoted character length.

    Raises
    ------
    ValueError
        If `smartirs` is not a string of length 3 or one of the decomposed value
        doesn't fit the list of permissible values.
    """
    if isinstance(smartirs, str) and re.match('...\\....', smartirs):
        match = re.match('(?P<ddd>...)\\.(?P<qqq>...)', smartirs)
        raise ValueError('The notation {ddd}.{qqq} specifies two term-weighting schemes, one for collection documents ({ddd}) and one for queries ({qqq}). You must train two separate tf-idf models.'.format(ddd=match.group('ddd'), qqq=match.group('qqq')))
    if not isinstance(smartirs, str) or len(smartirs) != 3:
        raise ValueError('Expected a string of length 3 got ' + smartirs)
    w_tf, w_df, w_n = smartirs
    if w_tf not in 'btnaldL':
        raise ValueError("Expected term frequency weight to be one of 'btnaldL', got {}".format(w_tf))
    if w_df not in 'xnftp':
        raise ValueError("Expected inverse document frequency weight to be one of 'xnftp', got {}".format(w_df))
    if w_n not in 'xncub':
        raise ValueError("Expected normalization weight to be one of 'xncub', got {}".format(w_n))
    if w_tf == 't':
        w_tf = 'n'
    if w_df == 'x':
        w_df = 'n'
    if w_n == 'x':
        w_n = 'n'
    return w_tf + w_df + w_n