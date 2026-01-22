import struct
from oslo_log import log as logging
def eat_chunk(self, chunk):
    """Call this to present chunks of the file to the inspector."""
    pre_regions = set(self._capture_regions.keys())
    self._total_count += len(chunk)
    self._capture(chunk)
    self.post_process()
    new_regions = set(self._capture_regions.keys()) - pre_regions
    if new_regions:
        self._capture(chunk, only=new_regions)