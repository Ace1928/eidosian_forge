import re
import sys
from pprint import pprint
def is_int_list(value, min=None, max=None):
    """
    Check that the value is a list of integers.
    
    You can optionally specify the minimum and maximum number of members.
    
    Each list member is checked that it is an integer.
    
    >>> vtor.check('int_list', ())
    []
    >>> vtor.check('int_list', [])
    []
    >>> vtor.check('int_list', (1, 2))
    [1, 2]
    >>> vtor.check('int_list', [1, 2])
    [1, 2]
    >>> vtor.check('int_list', [1, 'a'])
    Traceback (most recent call last):
    VdtTypeError: the value "a" is of the wrong type.
    """
    return [is_integer(mem) for mem in is_list(value, min, max)]