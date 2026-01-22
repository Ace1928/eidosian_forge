import contextlib
import re
import textwrap
from typing import Iterable, List, Tuple, Union
def do_wrap_field_lists(text: str, field_idx: List[Tuple[int, int]], lines: List[str], text_idx: int, indentation: str, wrap_length: int) -> Tuple[List[str], int]:
    """Wrap field lists in the long description.

    Parameters
    ----------
    text : str
        The long description text.
    field_idx : list
        The list of field list indices found in the description text.
    lines : list
        The list of formatted lines in the description that come before the
        first parameter list item.
    text_idx : int
        The index in the description of the end of the last parameter list
        item.
    indentation : str
        The string to use to indent each line in the long description.
    wrap_length : int
         The line length at which to wrap long lines in the description.

    Returns
    -------
    lines, text_idx : tuple
         A list of the long description lines and the index in the long
        description where the last parameter list item ended.
    """
    lines.extend(description_to_list(text[text_idx:field_idx[0][0]], indentation, wrap_length))
    for _idx, __ in enumerate(field_idx):
        _field_name = text[field_idx[_idx][0]:field_idx[_idx][1]]
        _field_body = _do_join_field_body(text, field_idx, _idx)
        if len(f'{_field_name}{_field_body}') <= wrap_length - len(indentation):
            _field = f'{_field_name}{_field_body}'
            lines.append(f'{indentation}{_field}')
        else:
            lines.extend(_do_wrap_field(_field_name, _field_body, indentation, wrap_length))
        text_idx = field_idx[_idx][1]
    return (lines, text_idx)