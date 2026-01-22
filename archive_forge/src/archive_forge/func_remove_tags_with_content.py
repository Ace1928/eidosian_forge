import re
from html.entities import name2codepoint
from typing import Iterable, Match, AnyStr, Optional, Pattern, Tuple, Union
from urllib.parse import urljoin
from w3lib.util import to_unicode
from w3lib.url import safe_url_string
from w3lib._types import StrOrBytes
def remove_tags_with_content(text: AnyStr, which_ones: Iterable[str]=(), encoding: Optional[str]=None) -> str:
    """Remove tags and their content.

    `which_ones` is a tuple of which tags to remove including their content.
    If is empty, returns the string unmodified.

    >>> import w3lib.html
    >>> doc = '<div><p><b>This is a link:</b> <a href="http://www.example.com">example</a></p></div>'
    >>> w3lib.html.remove_tags_with_content(doc, which_ones=('b',))
    '<div><p> <a href="http://www.example.com">example</a></p></div>'
    >>>

    """
    utext = to_unicode(text, encoding)
    if which_ones:
        tags = '|'.join([f'<{tag}\\b.*?</{tag}>|<{tag}\\s*/>' for tag in which_ones])
        retags = re.compile(tags, re.DOTALL | re.IGNORECASE)
        utext = retags.sub('', utext)
    return utext