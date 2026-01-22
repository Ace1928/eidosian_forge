import abc
from oslo_cache import core
from oslo_config import cfg
from heat.common import cache
def has_extension(self, alias):
    """Check if specific extension is present."""
    if self._extensions is None:
        self._extensions = set(self._list_extensions())
    return alias in self._extensions