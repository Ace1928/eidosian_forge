from __future__ import absolute_import, division, print_function
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def convert_to_aligned_bytes(self, size):
    """Convert size to the truncated byte size that aligns on the segment size."""
    size_bytes = int(size * self.SIZE_UNIT_MAP[self.size_unit])
    segment_size_bytes = int(self.segment_size_kb * self.SIZE_UNIT_MAP['kb'])
    segment_count = int(size_bytes / segment_size_bytes)
    return segment_count * segment_size_bytes