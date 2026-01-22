from unittest import TestCase
import simplejson as json
from operator import itemgetter
def test_item_sort_key_value(self):
    a = {'a': 1, 'b': 0}
    self.assertEqual('{"b": 0, "a": 1}', json.dumps(a, item_sort_key=lambda kv: kv[1]))