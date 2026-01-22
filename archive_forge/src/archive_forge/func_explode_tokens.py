from __future__ import unicode_literals
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.token import Token
def explode_tokens(tokenlist):
    """
    Turn a list of (token, text) tuples into another list where each string is
    exactly one character.

    It should be fine to call this function several times. Calling this on a
    list that is already exploded, is a null operation.

    :param tokenlist: List of (token, text) tuples.
    """
    if getattr(tokenlist, 'exploded', False):
        return tokenlist
    result = []
    for token, string in tokenlist:
        for c in string:
            result.append((token, c))
    return _ExplodedList(result)