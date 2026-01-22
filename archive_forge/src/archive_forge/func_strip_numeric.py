import re
import string
import glob
from gensim import utils
from gensim.parsing.porter import PorterStemmer
def strip_numeric(s):
    """Remove digits from `s` using :const:`~gensim.parsing.preprocessing.RE_NUMERIC`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode  string without digits.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.parsing.preprocessing import strip_numeric
        >>> strip_numeric("0text24gensim365test")
        u'textgensimtest'

    """
    s = utils.to_unicode(s)
    return RE_NUMERIC.sub('', s)