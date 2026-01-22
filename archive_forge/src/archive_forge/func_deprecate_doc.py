import functools
import inspect
import textwrap
import threading
import types
import warnings
from typing import TypeVar, Type, Callable, Any, Union
from inspect import signature
from functools import wraps
def deprecate_doc(old_doc, message):
    """
        Returns a given docstring with a deprecation message prepended
        to it.
        """
    if not old_doc:
        old_doc = ''
    old_doc = textwrap.dedent(old_doc).strip('\n')
    new_doc = '\n.. deprecated:: {since}\n    {message}\n\n'.format(**{'since': since, 'message': message.strip()}) + old_doc
    if not old_doc:
        new_doc += '\\ '
    return new_doc