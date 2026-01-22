import pathlib
import subprocess
import sys
import time
from collections import defaultdict
from functools import lru_cache
from ._parsing import LogCatcher, cvsecs, parse_ffmpeg_header
from ._utils import _popen_kwargs, get_ffmpeg_exe, logger
def get_compiled_h264_encoders():
    cmd = [get_ffmpeg_exe(), '-hide_banner', '-encoders']
    p = subprocess.run(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = p.stdout.decode().replace('\r', '')
    header_footer = stdout.split('------')
    footer = header_footer[1].strip('\n')
    encoders = []
    for line in footer.split('\n'):
        line = line.strip()
        encoder = line.split(' ')[1]
        if encoder in h264_encoder_preference:
            encoders.append(encoder)
        elif line[0] == 'V' and 'H.264' in line:
            encoders.append(encoder)
    encoders.sort(reverse=True, key=lambda x: h264_encoder_preference[x])
    if 'h264_nvenc' in encoders:
        for encoder in ['nvenc', 'nvenc_h264']:
            if encoder in encoders:
                encoders.remove(encoder)
    return tuple(encoders)