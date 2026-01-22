import pathlib
import subprocess
import sys
import time
from collections import defaultdict
from functools import lru_cache
from ._parsing import LogCatcher, cvsecs, parse_ffmpeg_header
from ._utils import _popen_kwargs, get_ffmpeg_exe, logger
@lru_cache()
def get_first_available_h264_encoder():
    compiled_encoders = get_compiled_h264_encoders()
    for encoder in compiled_encoders:
        if ffmpeg_test_encoder(encoder):
            return encoder
    else:
        raise RuntimeError('No valid H.264 encoder was found with the ffmpeg installation')