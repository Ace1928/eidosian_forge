import re
import string
import glob
from gensim import utils
from gensim.parsing.porter import PorterStemmer
def stem_text(text):
    """Transform `s` into lowercase and stem it.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Unicode lowercased and porter-stemmed version of string `text`.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.parsing.preprocessing import stem_text
        >>> stem_text("While it is quite useful to be able to search a large collection of documents almost instantly.")
        u'while it is quit us to be abl to search a larg collect of document almost instantly.'

    """
    text = utils.to_unicode(text)
    p = PorterStemmer()
    return ' '.join((p.stem(word) for word in text.split()))