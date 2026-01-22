import warnings
import sys
import socket
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from ..module import get_introspection_module
from .._gi import (variant_type_from_string, source_new,
from ..overrides import override, deprecated, deprecated_attr
from gi import PyGIDeprecationWarning, version_info
from gi import _option as option
from gi import _gi
from gi._error import GError
@classmethod
def split_signature(klass, signature):
    """Return a list of the element signatures of the topmost signature tuple.

        If the signature is not a tuple, it returns one element with the entire
        signature. If the signature is an empty tuple, the result is [].

        This is useful for e. g. iterating over method parameters which are
        passed as a single Variant.
        """
    if signature == '()':
        return []
    if not signature.startswith('('):
        return [signature]
    result = []
    head = ''
    tail = signature[1:-1]
    while tail:
        c = tail[0]
        head += c
        tail = tail[1:]
        if c in ('m', 'a'):
            continue
        if c in ('(', '{'):
            level = 1
            up = c
            if up == '(':
                down = ')'
            else:
                down = '}'
            while level > 0:
                c = tail[0]
                head += c
                tail = tail[1:]
                if c == up:
                    level += 1
                elif c == down:
                    level -= 1
        result.append(head)
        head = ''
    return result