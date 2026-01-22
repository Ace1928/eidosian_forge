import contextlib
import re
import textwrap
from typing import Iterable, List, Tuple, Union
def description_to_list(description: str, indentation: str, wrap_length: int) -> List[str]:
    """Convert the description to a list of wrap length lines.

    Parameters
    ----------
    description : str
        The docstring description.
    indentation : str
        The indentation (number of spaces or tabs) to place in front of each
        line.
    wrap_length : int
        The column to wrap each line at.

    Returns
    -------
    _wrapped_lines : list
          A list containing each line of the description wrapped at wrap_length.
    """
    if len(re.findall('\\n\\n', description)) <= 0:
        return textwrap.wrap(textwrap.dedent(description), width=wrap_length, initial_indent=indentation, subsequent_indent=indentation)
    _wrapped_lines = []
    for _line in description.split('\n\n'):
        _wrapped_line = textwrap.wrap(textwrap.dedent(_line), width=wrap_length, initial_indent=indentation, subsequent_indent=indentation)
        if _wrapped_line:
            _wrapped_lines.extend(_wrapped_line)
        _wrapped_lines.append('')
        with contextlib.suppress(IndexError):
            if not _wrapped_lines[-1] and (not _wrapped_lines[-2]):
                _wrapped_lines.pop(-1)
    if description[-len(indentation) - 1:-len(indentation)] == '\n' and description[-len(indentation) - 2:-len(indentation)] != '\n\n':
        _wrapped_lines.pop(-1)
    return _wrapped_lines