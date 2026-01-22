import collections
import pyrfc3339
from ._conditions import (
def parse_caveat(cav):
    """ Parses a caveat into an identifier, identifying the checker that should
    be used, and the argument to the checker (the rest of the string).

    The identifier is taken from all the characters before the first
    space character.
    :return two string, identifier and arg
    """
    if cav == '':
        raise ValueError('empty caveat')
    try:
        i = cav.index(' ')
    except ValueError:
        return (cav, '')
    if i == 0:
        raise ValueError('caveat starts with space character')
    return (cav[0:i], cav[i + 1:])