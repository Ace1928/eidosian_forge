from ..bisect_multi import bisect_multi_bytes
from . import TestCase
def test_change_direction_in_single_key_search(self):
    calls = []

    def missing_at_5(location_keys):
        calls.append(location_keys)
        result = []
        for location_key in location_keys:
            if location_key[0] == 5:
                result.append((location_key, False))
            elif location_key[0] > 5:
                result.append((location_key, -1))
            else:
                result.append((location_key, +1))
        return result
    self.assertEqual([], list(bisect_multi_bytes(missing_at_5, 8, ['foo', 'bar'])))
    self.assertEqual([[(4, 'foo'), (4, 'bar')], [(6, 'foo'), (6, 'bar')], [(5, 'foo'), (5, 'bar')]], calls)