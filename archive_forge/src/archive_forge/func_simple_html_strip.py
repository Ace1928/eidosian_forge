import re
import itertools
import textwrap
import functools
from jaraco.functools import compose, method_cache
from jaraco.context import ExceptionTrap
def simple_html_strip(s):
    """
    Remove HTML from the string `s`.

    >>> str(simple_html_strip(''))
    ''

    >>> print(simple_html_strip('A <bold>stormy</bold> day in paradise'))
    A stormy day in paradise

    >>> print(simple_html_strip('Somebody <!-- do not --> tell the truth.'))
    Somebody  tell the truth.

    >>> print(simple_html_strip('What about<br/>\\nmultiple lines?'))
    What about
    multiple lines?
    """
    html_stripper = re.compile('(<!--.*?-->)|(<[^>]*>)|([^<]+)', re.DOTALL)
    texts = (match.group(3) or '' for match in html_stripper.finditer(s))
    return ''.join(texts)