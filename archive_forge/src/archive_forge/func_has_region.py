import struct
from oslo_log import log as logging
def has_region(self, name):
    """Returns True if named region has been defined."""
    return name in self._capture_regions