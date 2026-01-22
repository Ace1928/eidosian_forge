import aifc
import audioop
import io
import os
import platform
import stat
import subprocess
import sys
import wave
def get_flac_data(self, convert_rate=None, convert_width=None):
    """
        Returns a byte string representing the contents of a FLAC file containing the audio represented by the ``AudioData`` instance.

        Note that 32-bit FLAC is not supported. If the audio data is 32-bit and ``convert_width`` is not specified, then the resulting FLAC will be a 24-bit FLAC.

        If ``convert_rate`` is specified and the audio sample rate is not ``convert_rate`` Hz, the resulting audio is resampled to match.

        If ``convert_width`` is specified and the audio samples are not ``convert_width`` bytes each, the resulting audio is converted to match.

        Writing these bytes directly to a file results in a valid `FLAC file <https://en.wikipedia.org/wiki/FLAC>`__.
        """
    assert convert_width is None or (convert_width % 1 == 0 and 1 <= convert_width <= 3), 'Sample width to convert to must be between 1 and 3 inclusive'
    if self.sample_width > 3 and convert_width is None:
        convert_width = 3
    wav_data = self.get_wav_data(convert_rate, convert_width)
    flac_converter = get_flac_converter()
    if os.name == 'nt':
        startup_info = subprocess.STARTUPINFO()
        startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup_info.wShowWindow = subprocess.SW_HIDE
    else:
        startup_info = None
    process = subprocess.Popen([flac_converter, '--stdout', '--totally-silent', '--best', '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, startupinfo=startup_info)
    flac_data, stderr = process.communicate(wav_data)
    return flac_data