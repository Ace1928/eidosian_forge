from __future__ import division
import array
import os
import subprocess
from tempfile import TemporaryFile, NamedTemporaryFile
import wave
import sys
import struct
from .logging_utils import log_conversion, log_subprocess_output
from .utils import mediainfo_json, fsdecode
import base64
from collections import namedtuple
from io import BytesIO
from .utils import (
from .exceptions import (
from . import effects
@classmethod
def from_file_using_temporary_files(cls, file, format=None, codec=None, parameters=None, start_second=None, duration=None, **kwargs):
    orig_file = file
    file, close_file = _fd_or_path_or_tempfile(file, 'rb', tempfile=False)
    if format:
        format = format.lower()
        format = AUDIO_FILE_EXT_ALIASES.get(format, format)

    def is_format(f):
        f = f.lower()
        if format == f:
            return True
        if isinstance(orig_file, basestring):
            return orig_file.lower().endswith('.{0}'.format(f))
        if isinstance(orig_file, bytes):
            return orig_file.lower().endswith('.{0}'.format(f).encode('utf8'))
        return False
    if is_format('wav'):
        try:
            obj = cls._from_safe_wav(file)
            if close_file:
                file.close()
            if start_second is None and duration is None:
                return obj
            elif start_second is not None and duration is None:
                return obj[start_second * 1000:]
            elif start_second is None and duration is not None:
                return obj[:duration * 1000]
            else:
                return obj[start_second * 1000:(start_second + duration) * 1000]
        except:
            file.seek(0)
    elif is_format('raw') or is_format('pcm'):
        sample_width = kwargs['sample_width']
        frame_rate = kwargs['frame_rate']
        channels = kwargs['channels']
        metadata = {'sample_width': sample_width, 'frame_rate': frame_rate, 'channels': channels, 'frame_width': channels * sample_width}
        obj = cls(data=file.read(), metadata=metadata)
        if close_file:
            file.close()
        if start_second is None and duration is None:
            return obj
        elif start_second is not None and duration is None:
            return obj[start_second * 1000:]
        elif start_second is None and duration is not None:
            return obj[:duration * 1000]
        else:
            return obj[start_second * 1000:(start_second + duration) * 1000]
    input_file = NamedTemporaryFile(mode='wb', delete=False)
    try:
        input_file.write(file.read())
    except OSError:
        input_file.flush()
        input_file.close()
        input_file = NamedTemporaryFile(mode='wb', delete=False, buffering=2 ** 31 - 1)
        if close_file:
            file.close()
        close_file = True
        file = open(orig_file, buffering=2 ** 13 - 1, mode='rb')
        reader = file.read(2 ** 31 - 1)
        while reader:
            input_file.write(reader)
            reader = file.read(2 ** 31 - 1)
    input_file.flush()
    if close_file:
        file.close()
    output = NamedTemporaryFile(mode='rb', delete=False)
    conversion_command = [cls.converter, '-y']
    if format:
        conversion_command += ['-f', format]
    if codec:
        conversion_command += ['-acodec', codec]
    conversion_command += ['-i', input_file.name, '-vn', '-f', 'wav']
    if start_second is not None:
        conversion_command += ['-ss', str(start_second)]
    if duration is not None:
        conversion_command += ['-t', str(duration)]
    conversion_command += [output.name]
    if parameters is not None:
        conversion_command.extend(parameters)
    log_conversion(conversion_command)
    with open(os.devnull, 'rb') as devnull:
        p = subprocess.Popen(conversion_command, stdin=devnull, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p_out, p_err = p.communicate()
    log_subprocess_output(p_out)
    log_subprocess_output(p_err)
    try:
        if p.returncode != 0:
            raise CouldntDecodeError('Decoding failed. ffmpeg returned error code: {0}\n\nOutput from ffmpeg/avlib:\n\n{1}'.format(p.returncode, p_err.decode(errors='ignore')))
        obj = cls._from_safe_wav(output)
    finally:
        input_file.close()
        output.close()
        os.unlink(input_file.name)
        os.unlink(output.name)
    if start_second is None and duration is None:
        return obj
    elif start_second is not None and duration is None:
        return obj[0:]
    elif start_second is None and duration is not None:
        return obj[:duration * 1000]
    else:
        return obj[0:duration * 1000]