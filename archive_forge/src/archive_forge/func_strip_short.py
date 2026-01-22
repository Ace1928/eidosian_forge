import re
import string
import glob
from gensim import utils
from gensim.parsing.porter import PorterStemmer
def strip_short(s, minsize=3):
    """Remove words with length lesser than `minsize` from `s`.

    Parameters
    ----------
    s : str
    minsize : int, optional

    Returns
    -------
    str
        Unicode string without short words.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.parsing.preprocessing import strip_short
        >>> strip_short("salut les amis du 59")
        u'salut les amis'
        >>>
        >>> strip_short("one two three four five six seven eight nine ten", minsize=5)
        u'three seven eight'

    """
    s = utils.to_unicode(s)
    return ' '.join(remove_short_tokens(s.split(), minsize))