from ..bisect_multi import bisect_multi_bytes
from . import TestCase
def missing_bar_at_1_foo_at_3(location_keys):
    calls.append(location_keys)
    result = []
    for location_key in location_keys:
        if location_key[1] == 'bar':
            if location_key[0] == 1:
                result.append((location_key, False))
            else:
                result.append((location_key, -1))
        elif location_key[1] == 'foo':
            if location_key[0] == 3:
                result.append((location_key, False))
            else:
                result.append((location_key, +1))
    return result