import codecs
from html.entities import codepoint2name
from html.entities import name2codepoint
import re
from urllib.parse import quote_plus
import markupsafe
def htmlentityreplace_errors(ex):
    """An encoding error handler.

    This python codecs error handler replaces unencodable
    characters with HTML entities, or, if no HTML entity exists for
    the character, XML character references::

        >>> 'The cost was €12.'.encode('latin1', 'htmlentityreplace')
        'The cost was &euro;12.'
    """
    if isinstance(ex, UnicodeEncodeError):
        bad_text = ex.object[ex.start:ex.end]
        text = _html_entities_escaper.escape(bad_text)
        return (str(text), ex.end)
    raise ex