import re
import string
import glob
from gensim import utils
from gensim.parsing.porter import PorterStemmer
def split_on_space(s):
    """Split line by spaces, used in :class:`gensim.corpora.lowcorpus.LowCorpus`.

    Parameters
    ----------
    s : str
        Some line.

    Returns
    -------
    list of str
        List of tokens from `s`.

    """
    return [word for word in utils.to_unicode(s).strip().split(' ') if word]