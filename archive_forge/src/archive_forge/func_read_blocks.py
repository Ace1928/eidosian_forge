import mad
from . import DecodeError
from .base import AudioFile
def read_blocks(self, block_size=4096):
    """Generates buffers containing PCM data for the audio file.
        """
    while True:
        out = self.mf.read(block_size)
        if not out:
            break
        yield bytes(out)