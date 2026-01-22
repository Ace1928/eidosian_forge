from collections import namedtuple
import warnings
def setnframes(self, nframes):
    if self._nframeswritten:
        raise Error('cannot change parameters after starting to write')
    if nframes < 0:
        raise Error('# of frames cannot be negative')
    self._nframes = nframes