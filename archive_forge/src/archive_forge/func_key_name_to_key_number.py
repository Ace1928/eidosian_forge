import numpy as np
import re
from .constants import DRUM_MAP, INSTRUMENT_MAP, INSTRUMENT_CLASSES
def key_name_to_key_number(key_string):
    """Convert a key name string to key number.

    Parameters
    ----------
    key_string : str
        Format is ``'(root) (mode)'``, where:
          * ``(root)`` is one of ABCDEFG or abcdefg.  A lowercase root
            indicates a minor key when no mode string is specified.  Optionally
            a # for sharp or b for flat can be specified.

          * ``(mode)`` is optionally specified either as one of 'M', 'Maj',
            'Major', 'maj', or 'major' for major or 'm', 'Min', 'Minor', 'min',
            'minor' for minor.  If no mode is specified and the root is
            uppercase, the mode is assumed to be major; if the root is
            lowercase, the mode is assumed to be minor.

    Returns
    -------
    key_number : int
        Integer representing the key and its mode.  Integers from 0 to 11
        represent major keys from C to B; 12 to 23 represent minor keys from C
        to B.
    """
    major_strs = ['M', 'Maj', 'Major', 'maj', 'major']
    minor_strs = ['m', 'Min', 'Minor', 'min', 'minor']
    pattern = re.compile('^(?P<key>[ABCDEFGabcdefg])(?P<flatsharp>[#b]?) ?(?P<mode>(?:(?:' + ')|(?:'.join(major_strs + minor_strs) + '))?)$')
    result = re.match(pattern, key_string)
    if result is None:
        raise ValueError('Supplied key {} is not valid.'.format(key_string))
    result = result.groupdict()
    key_number = {'c': 0, 'd': 2, 'e': 4, 'f': 5, 'g': 7, 'a': 9, 'b': 11}[result['key'].lower()]
    if result['flatsharp']:
        if result['flatsharp'] == '#':
            key_number += 1
        elif result['flatsharp'] == 'b':
            key_number -= 1
    key_number = key_number % 12
    if result['mode'] in minor_strs or (result['key'].islower() and result['mode'] not in major_strs):
        key_number += 12
    return key_number