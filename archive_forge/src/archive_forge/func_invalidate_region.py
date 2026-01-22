import secrets
from dogpile.cache import region
from dogpile.cache import util
from oslo_cache import core as cache
from keystone.common.cache import _context_cache
import keystone.conf
def invalidate_region(self):
    new_region_id = self._generate_new_id()
    self._invalidation_region.set(self._region_key, new_region_id)
    return new_region_id