from __future__ import (absolute_import, division, print_function)
import re
def find_current_values(reorder_current, reorder_filtered):
    """Find keyvalues in current according to keys from filtered"""
    result = {}
    for key, value in reorder_filtered.items():
        if isinstance(value, dict):
            result[key] = find_current_values(reorder_current[key], value)
        elif isinstance(value, list):
            result[key] = []
            for i in range(len(value)):
                if isinstance(value[i], dict):
                    result[key].append(find_current_values(reorder_current[key][i], value[i]))
                else:
                    result[key].append(reorder_current[key])
        elif isinstance(value, str):
            result[key] = reorder_current[key]
    return result