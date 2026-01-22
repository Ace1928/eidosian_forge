from __future__ import unicode_literals
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.token import Token
def token_list_len(tokenlist):
    """
    Return the amount of characters in this token list.

    :param tokenlist: List of (token, text) or (token, text, mouse_handler)
                      tuples.
    """
    ZeroWidthEscape = Token.ZeroWidthEscape
    return sum((len(item[1]) for item in tokenlist if item[0] != ZeroWidthEscape))