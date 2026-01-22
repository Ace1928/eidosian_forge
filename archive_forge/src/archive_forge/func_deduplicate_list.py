from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.six import string_types
def deduplicate_list(original_list):
    """
    Creates a deduplicated list with the order in which each item is first found.
    """
    seen = set()
    return [x for x in original_list if x not in seen and (not seen.add(x))]