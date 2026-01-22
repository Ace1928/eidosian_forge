from __future__ import (absolute_import, division, print_function)
from stringprep import (
from unicodedata import normalize
from ansible.module_utils.six import text_type
def is_ral_string(string):
    """RFC3454 Check bidirectional category of the string"""
    if in_table_d1(string[0]):
        if not in_table_d1(string[-1]):
            raise ValueError('RFC3454: incorrect bidirectional RandALCat string.')
        return True
    return False