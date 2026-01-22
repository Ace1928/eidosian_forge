import secrets
from dogpile.cache import region
from dogpile.cache import util
from oslo_cache import core as cache
from keystone.common.cache import _context_cache
import keystone.conf
def configure_cache(region=None):
    if region is None:
        region = CACHE_REGION
    configured = region.is_configured
    cache.configure_cache_region(CONF, region)
    if not configured:
        region.wrap(_context_cache._ResponseCacheProxy)
        region_manager = RegionInvalidationManager(CACHE_INVALIDATION_REGION, region.name)
        region.key_mangler = key_mangler_factory(region_manager, region.key_mangler)
        region.region_invalidator = DistributedInvalidationStrategy(region_manager)