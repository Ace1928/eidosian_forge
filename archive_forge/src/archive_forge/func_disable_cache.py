import time
from . import debug, errors, osutils, revision, trace
def disable_cache(self):
    """Disable and clear the cache."""
    self._cache = None
    self._cache_misses = None
    self.missing_keys = set()