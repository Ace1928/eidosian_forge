from dogpile.cache import region as dp_region
from oslo_cache import core
from oslo_cache.tests import test_cache
from oslo_config import fixture as config_fixture
from oslo_utils import fixture as time_fixture
def test_dict_backend_clear_cache(self):
    self.region.set(KEY, VALUE)
    self.time_fixture.advance_time_seconds(1)
    self.assertEqual(1, len(self.region.backend.cache))
    self.region.backend._clear()
    self.assertEqual(0, len(self.region.backend.cache))