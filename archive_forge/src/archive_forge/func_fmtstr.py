import re
from cwcwidth import wcswidth, wcwidth
from itertools import chain
from typing import (
from .escseqparse import parse, remove_ansi
from .termformatconstants import (
def fmtstr(string: Union[str, FmtStr], *args: Any, **kwargs: Any) -> FmtStr:
    """
    Convenience function for creating a FmtStr

    >>> fmtstr('asdf', 'blue', 'on_red', 'bold')
    on_red(bold(blue('asdf')))
    >>> fmtstr('blarg', fg='blue', bg='red', bold=True)
    on_red(bold(blue('blarg')))
    """
    atts = parse_args(args, kwargs)
    if isinstance(string, str):
        string = FmtStr.from_str(string)
    elif not isinstance(string, FmtStr):
        raise ValueError(f'Bad Args: {string!r} (of type {type(string)}), {args!r}, {kwargs!r}')
    return string.copy_with_new_atts(**atts)