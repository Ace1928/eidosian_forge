import re
from six.moves import html_entities as entities
import six
def plaintext(text, keeplinebreaks=True):
    """Return the text with all entities and tags removed.
    
    >>> plaintext('<b>1 &lt; 2</b>')
    '1 < 2'
    
    The `keeplinebreaks` parameter can be set to ``False`` to replace any line
    breaks by simple spaces:
    
    >>> plaintext('''<b>1
    ... &lt;
    ... 2</b>''', keeplinebreaks=False)
    '1 < 2'
    
    :param text: the text to convert to plain text
    :param keeplinebreaks: whether line breaks in the text should be kept intact
    :return: the text with tags and entities removed
    """
    text = stripentities(striptags(text))
    if not keeplinebreaks:
        text = text.replace('\n', ' ')
    return text