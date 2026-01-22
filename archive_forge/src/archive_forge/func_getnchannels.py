from collections import namedtuple
import warnings
def getnchannels(self):
    if not self._nchannels:
        raise Error('number of channels not set')
    return self._nchannels