from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
def is_entry_present(self, cache_item, entry):
    for item in cache_item:
        if item[0] == entry[0]:
            return True
    return False