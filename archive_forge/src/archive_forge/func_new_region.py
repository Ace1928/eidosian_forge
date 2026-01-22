import struct
from oslo_log import log as logging
def new_region(self, name, region):
    """Add a new CaptureRegion by name."""
    if self.has_region(name):
        raise ImageFormatError('Inspector re-added region %s' % name)
    self._capture_regions[name] = region