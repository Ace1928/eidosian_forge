from collections import namedtuple
import warnings
def setframerate(self, framerate):
    if self._nframeswritten:
        raise Error('cannot change parameters after starting to write')
    self._framerate = framerate