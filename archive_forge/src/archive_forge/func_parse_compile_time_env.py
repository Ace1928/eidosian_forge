from __future__ import absolute_import
import os
from .. import Utils
def parse_compile_time_env(s, current_settings=None):
    """
    Parses a comma-separated list of pragma options. Whitespace
    is not considered.

    >>> parse_compile_time_env('      ')
    {}
    >>> (parse_compile_time_env('HAVE_OPENMP=True') ==
    ... {'HAVE_OPENMP': True})
    True
    >>> parse_compile_time_env('  asdf')
    Traceback (most recent call last):
       ...
    ValueError: Expected "=" in option "asdf"
    >>> parse_compile_time_env('NUM_THREADS=4') == {'NUM_THREADS': 4}
    True
    >>> parse_compile_time_env('unknown=anything') == {'unknown': 'anything'}
    True
    """
    if current_settings is None:
        result = {}
    else:
        result = current_settings
    for item in s.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError('Expected "=" in option "%s"' % item)
        name, value = [s.strip() for s in item.split('=', 1)]
        result[name] = parse_variable_value(value)
    return result