from collections import namedtuple
import warnings
def setnchannels(self, nchannels):
    if self._nframeswritten:
        raise Error('cannot change parameters after starting to write')
    if nchannels not in (1, 2, 4):
        raise Error('only 1, 2, or 4 channels supported')
    self._nchannels = nchannels