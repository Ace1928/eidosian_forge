from __future__ import unicode_literals
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.token import Token
def token_list_to_text(tokenlist):
    """
    Concatenate all the text parts again.
    """
    ZeroWidthEscape = Token.ZeroWidthEscape
    return ''.join((item[1] for item in tokenlist if item[0] != ZeroWidthEscape))