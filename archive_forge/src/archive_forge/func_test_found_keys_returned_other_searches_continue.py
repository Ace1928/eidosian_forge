from ..bisect_multi import bisect_multi_bytes
from . import TestCase
def test_found_keys_returned_other_searches_continue(self):
    calls = []

    def find_bar_at_1_foo_missing_at_0(location_keys):
        calls.append(location_keys)
        result = []
        for location_key in location_keys:
            if location_key == (1, 'bar'):
                result.append((location_key, 'bar-result'))
            elif location_key[0] == 0:
                result.append((location_key, False))
            else:
                result.append((location_key, -1))
        return result
    self.assertEqual([('bar', 'bar-result')], list(bisect_multi_bytes(find_bar_at_1_foo_missing_at_0, 4, ['foo', 'bar'])))
    self.assertEqual([[(2, 'foo'), (2, 'bar')], [(1, 'foo'), (1, 'bar')], [(0, 'foo')]], calls)