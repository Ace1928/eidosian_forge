from __future__ import absolute_import
import os
from .. import Utils
def parse_directive_value(name, value, relaxed_bool=False):
    """
    Parses value as an option value for the given name and returns
    the interpreted value. None is returned if the option does not exist.

    >>> print(parse_directive_value('nonexisting', 'asdf asdfd'))
    None
    >>> parse_directive_value('boundscheck', 'True')
    True
    >>> parse_directive_value('boundscheck', 'true')
    Traceback (most recent call last):
       ...
    ValueError: boundscheck directive must be set to True or False, got 'true'

    >>> parse_directive_value('c_string_encoding', 'us-ascii')
    'ascii'
    >>> parse_directive_value('c_string_type', 'str')
    'str'
    >>> parse_directive_value('c_string_type', 'bytes')
    'bytes'
    >>> parse_directive_value('c_string_type', 'bytearray')
    'bytearray'
    >>> parse_directive_value('c_string_type', 'unicode')
    'unicode'
    >>> parse_directive_value('c_string_type', 'unnicode')
    Traceback (most recent call last):
    ValueError: c_string_type directive must be one of ('bytes', 'bytearray', 'str', 'unicode'), got 'unnicode'
    """
    type = directive_types.get(name)
    if not type:
        return None
    orig_value = value
    if type is bool:
        value = str(value)
        if value == 'True':
            return True
        if value == 'False':
            return False
        if relaxed_bool:
            value = value.lower()
            if value in ('true', 'yes'):
                return True
            elif value in ('false', 'no'):
                return False
        raise ValueError("%s directive must be set to True or False, got '%s'" % (name, orig_value))
    elif type is int:
        try:
            return int(value)
        except ValueError:
            raise ValueError("%s directive must be set to an integer, got '%s'" % (name, orig_value))
    elif type is str:
        return str(value)
    elif callable(type):
        return type(name, value)
    else:
        assert False