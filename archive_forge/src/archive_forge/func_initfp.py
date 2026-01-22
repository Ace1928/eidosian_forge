from collections import namedtuple
import warnings
def initfp(self, file):
    self._file = file
    self._framerate = 0
    self._nchannels = 0
    self._sampwidth = 0
    self._framesize = 0
    self._nframes = AUDIO_UNKNOWN_SIZE
    self._nframeswritten = 0
    self._datawritten = 0
    self._datalength = 0
    self._info = b''
    self._comptype = 'ULAW'