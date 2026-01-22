from struct import unpack
def read_array_start(self):
    """Arrays are encoded as a series of blocks."""
    self._block_count = self.read_long()