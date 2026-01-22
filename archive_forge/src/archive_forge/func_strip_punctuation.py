import re
import string
import glob
from gensim import utils
from gensim.parsing.porter import PorterStemmer
def strip_punctuation(s):
    """Replace ASCII punctuation characters with spaces in `s` using :const:`~gensim.parsing.preprocessing.RE_PUNCT`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string without punctuation characters.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.parsing.preprocessing import strip_punctuation
        >>> strip_punctuation("A semicolon is a stronger break than a comma, but not as much as a full stop!")
        u'A semicolon is a stronger break than a comma  but not as much as a full stop '

    """
    s = utils.to_unicode(s)
    return RE_PUNCT.sub(' ', s)