from typing import Iterable, List, Tuple
from triad.constants import TRIAD_VAR_QUOTE
from .assertion import assert_or_throw
from .string import validate_triad_var_name
def unquote_name(name: str, quote: str=TRIAD_VAR_QUOTE) -> str:
    """If the input is quoted, then get the inner string,
    otherwise do nothing.

    :param name: the name string
    :param quote: the quote char, defaults to `
    :return: the value without `
    """
    if validate_triad_var_name(name):
        return name
    if len(name) >= 2 and name[0] == name[-1] == quote:
        return name[1:-1].replace(quote + quote, quote)
    name = name.strip()
    assert_or_throw(len(name) > 0, ValueError('empty string is invalid'))
    return name