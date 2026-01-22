import breezy.errors as errors
import breezy.transactions as transactions
from breezy.tests import TestCase
def test_small_cache_with_references(self):
    weave = 'a weave'
    weave2 = 'another weave'
    self.transaction.map.add_weave('id', weave)
    self.transaction.map.add_weave('id2', weave2)
    self.assertEqual(weave, self.transaction.map.find_weave('id'))
    self.assertEqual(weave2, self.transaction.map.find_weave('id2'))
    weave = None
    self.assertEqual('a weave', self.transaction.map.find_weave('id'))