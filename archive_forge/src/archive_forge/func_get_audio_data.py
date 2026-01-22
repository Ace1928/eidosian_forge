from ctypes import memmove, byref, c_uint32, sizeof, cast, c_void_p, create_string_buffer, POINTER, c_char, \
from pyglet.libs.darwin import cf, CFSTR
from pyglet.libs.darwin.coreaudio import kCFURLPOSIXPathStyle, AudioStreamBasicDescription, ca, ExtAudioFileRef, \
from pyglet.media import StreamingSource, StaticSource
from pyglet.media.codecs import AudioFormat, MediaDecoder, AudioData
def get_audio_data(self, num_bytes, compensation_time=0.0):
    num_frames = c_uint32(num_bytes // self.convert_desc.mBytesPerFrame)
    if not self._bl:
        buffer = create_string_buffer(num_bytes)
        self._bl = AudioBufferList()
        self._bl.mNumberBuffers = 1
        self._bl.mBuffers[0].mNumberChannels = self.convert_desc.mChannelsPerFrame
        self._bl.mBuffers[0].mDataByteSize = num_bytes
        self._bl.mBuffers[0].mData = cast(buffer, c_void_p)
    while True:
        ca.ExtAudioFileRead(self._audref, byref(num_frames), byref(self._bl))
        size = self._bl.mBuffers[0].mDataByteSize
        if not size:
            break
        data = cast(self._bl.mBuffers[0].mData, POINTER(c_char))
        slice = data[:size]
        return AudioData(slice, size, 0.0, size / self.audio_format.sample_rate, [])
    return None