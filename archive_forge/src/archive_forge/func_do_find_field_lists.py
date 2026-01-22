import contextlib
import re
import textwrap
from typing import Iterable, List, Tuple, Union
def do_find_field_lists(text: str, style: str):
    """Determine if docstring contains any field lists.

    Parameters
    ----------
    text : str
        The docstring description to check for field list patterns.
    style : str
        The field list style used.

    Returns
    -------
    _field_idx, _wrap_parameters : tuple
        A list of tuples with each tuple containing the starting and ending
        position of each field list found in the passed description.
        A boolean indicating whether long field list lines should be wrapped.
    """
    _field_idx = []
    _wrap_parameters = False
    if style == 'epytext':
        _field_idx = [(_field.start(0), _field.end(0)) for _field in re.finditer(EPYTEXT_REGEX, text)]
        _wrap_parameters = True
    elif style == 'sphinx':
        _field_idx = [(_field.start(0), _field.end(0)) for _field in re.finditer(SPHINX_REGEX, text)]
        _wrap_parameters = True
    return (_field_idx, _wrap_parameters)