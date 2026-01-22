import re
import string
import glob
from gensim import utils
from gensim.parsing.porter import PorterStemmer
def lower_to_unicode(text, encoding='utf8', errors='strict'):
    """Lowercase `text` and convert to unicode, using :func:`gensim.utils.any2unicode`.

    Parameters
    ----------
    text : str
        Input text.
    encoding : str, optional
        Encoding that will be used for conversion.
    errors : str, optional
        Error handling behaviour, used as parameter for `unicode` function (python2 only).

    Returns
    -------
    str
        Unicode version of `text`.

    See Also
    --------
    :func:`gensim.utils.any2unicode`
        Convert any string to unicode-string.

    """
    return utils.to_unicode(text.lower(), encoding, errors)