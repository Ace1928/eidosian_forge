import collections
from ._caveat import error_caveat
from ._utils import condition_with_prefix
def is_valid_prefix(prefix):
    """Reports if prefix is valid.

    It must not contain white space or semi-colon.
    :param prefix string
    :return bool
    """
    return prefix.find(' ') == -1 and prefix.find(':') == -1