from dogpile.cache import region as dp_region
from oslo_cache import core
from oslo_cache.tests import test_cache
from oslo_config import fixture as config_fixture
from oslo_utils import fixture as time_fixture
def test_dict_backend_multi_keys_in_one_call(self):
    single_value = 'Test Value'
    single_key = 'testkey'
    multi_values = {'key1': 1, 'key2': 2, 'key3': 3}
    self.region.set(single_key, single_value)
    self.assertEqual(single_value, self.region.get(single_key))
    self.region.delete(single_key)
    self.assertEqual(NO_VALUE, self.region.get(single_key))
    self.region.set_multi(multi_values)
    cached_values = self.region.get_multi(multi_values.keys())
    for value in multi_values.values():
        self.assertIn(value, cached_values)
    self.assertEqual(len(multi_values.values()), len(cached_values))
    self.region.delete_multi(multi_values.keys())
    for value in self.region.get_multi(multi_values.keys()):
        self.assertEqual(NO_VALUE, value)