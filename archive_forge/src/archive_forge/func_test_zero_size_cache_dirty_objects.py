import breezy.errors as errors
import breezy.transactions as transactions
from breezy.tests import TestCase
def test_zero_size_cache_dirty_objects(self):
    self.transaction.set_cache_size(0)
    weave = DummyWeave('a weave')
    self.transaction.map.add_weave('id', weave)
    self.assertEqual(weave, self.transaction.map.find_weave('id'))
    weave = None
    self.transaction.register_dirty(self.transaction.map.find_weave('id'))
    self.assertNotEqual(None, self.transaction.map.find_weave('id'))