import __main__
import abc
import glob
import itertools
import keyword
import logging
import os
import re
import rlcompleter
import builtins
from enum import Enum
from typing import (
from . import inspection
from . import line as lineparts
from .line import LinePart
from .lazyre import LazyReCompile
from .simpleeval import safe_eval, evaluate_current_expression, EvaluationError
from .importcompletion import ModuleGatherer
def get_completer(completers: Sequence[BaseCompletionType], cursor_offset: int, line: str, *, locals_: Optional[Dict[str, Any]]=None, argspec: Optional[inspection.FuncProps]=None, history: Optional[List[str]]=None, current_block: Optional[str]=None, complete_magic_methods: Optional[bool]=None) -> Tuple[List[str], Optional[BaseCompletionType]]:
    """Returns a list of matches and an applicable completer

    If no matches available, returns a tuple of an empty list and None

    cursor_offset is the current cursor column
    line is a string of the current line
    kwargs (all optional):
        locals_ is a dictionary of the environment
        argspec is an inspection.FuncProps instance for the current function where
            the cursor is
        current_block is the possibly multiline not-yet-evaluated block of
            code which the current line is part of
        complete_magic_methods is a bool of whether we ought to complete
            double underscore methods like __len__ in method signatures
    """
    for completer in completers:
        try:
            matches = completer.matches(cursor_offset, line, locals_=locals_, funcprops=argspec, history=history, current_block=current_block, complete_magic_methods=complete_magic_methods)
        except Exception as e:
            logger.debug('Completer %r failed with unhandled exception: %s', completer, e)
            continue
        if matches is not None:
            return (sorted(matches), completer if matches else None)
    return ([], None)