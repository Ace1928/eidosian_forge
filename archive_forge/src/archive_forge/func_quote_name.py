from typing import Iterable, List, Tuple
from triad.constants import TRIAD_VAR_QUOTE
from .assertion import assert_or_throw
from .string import validate_triad_var_name
def quote_name(name: str, quote: str=TRIAD_VAR_QUOTE) -> str:
    """Add quote ` for strings that are not a valid triad var name.

    :param name: the name string
    :param quote: the quote char, defaults to `
    :return: the quoted(if necessary) string
    """
    if validate_triad_var_name(name):
        return name
    return quote + name.replace(quote, quote + quote) + quote