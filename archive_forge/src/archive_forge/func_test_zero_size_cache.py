import breezy.errors as errors
import breezy.transactions as transactions
from breezy.tests import TestCase
def test_zero_size_cache(self):
    self.transaction.set_cache_size(0)
    weave = DummyWeave('a weave')
    self.transaction.map.add_weave('id', weave)
    self.assertEqual(weave, self.transaction.map.find_weave('id'))
    weave = None
    self.transaction.register_clean(self.transaction.map.find_weave('id'))
    self.assertEqual(None, self.transaction.map.find_weave('id'))
    weave = DummyWeave('another weave')
    self.transaction.map.add_weave('id', weave)
    self.transaction.register_clean(self.transaction.map.find_weave('id'))
    self.assertEqual(weave, self.transaction.map.find_weave('id'))
    del weave
    self.assertEqual(DummyWeave('another weave'), self.transaction.map.find_weave('id'))