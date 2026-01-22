import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def split_words(text):
    """ Splits some text into words. Includes trailing whitespace
    on each word when appropriate.  """
    if not text or not text.strip():
        return []
    words = split_words_re.findall(text)
    return words