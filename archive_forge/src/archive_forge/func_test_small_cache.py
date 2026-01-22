import breezy.errors as errors
import breezy.transactions as transactions
from breezy.tests import TestCase
def test_small_cache(self):
    self.transaction.set_cache_size(1)
    self.transaction.map.add_weave('id', DummyWeave('a weave'))
    self.transaction.register_clean(self.transaction.map.find_weave('id'))
    self.assertEqual(DummyWeave('a weave'), self.transaction.map.find_weave('id'))
    self.transaction.map.add_weave('id2', DummyWeave('a weave also'))
    self.transaction.register_clean(self.transaction.map.find_weave('id2'))
    self.assertEqual(None, self.transaction.map.find_weave('id'))
    self.assertEqual(DummyWeave('a weave also'), self.transaction.map.find_weave('id2'))