from collections import namedtuple
import warnings
def getsampwidth(self):
    if not self._framerate:
        raise Error('sample width not specified')
    return self._sampwidth