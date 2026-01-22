from struct import unpack
def read_map_start(self):
    """Maps are encoded as a series of blocks."""
    self._block_count = self.read_long()