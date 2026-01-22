from collections import namedtuple
import warnings
def readframes(self, nframes):
    if self._encoding in _simple_encodings:
        if nframes == AUDIO_UNKNOWN_SIZE:
            data = self._file.read()
        else:
            data = self._file.read(nframes * self._framesize)
        self._soundpos += len(data) // self._framesize
        if self._encoding == AUDIO_FILE_ENCODING_MULAW_8:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=DeprecationWarning)
                import audioop
            data = audioop.ulaw2lin(data, self._sampwidth)
        return data
    return None