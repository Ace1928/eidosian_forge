import logging
import os
import subprocess
import sys
from functools import lru_cache
from pkg_resources import resource_filename
from ._definitions import FNAME_PER_PLATFORM, get_platform
def get_ffmpeg_exe():
    """
    Get the ffmpeg executable file. This can be the binary defined by
    the IMAGEIO_FFMPEG_EXE environment variable, the binary distributed
    with imageio-ffmpeg, an ffmpeg binary installed with conda, or the
    system ffmpeg (in that order). A RuntimeError is raised if no valid
    ffmpeg could be found.
    """
    exe = os.getenv('IMAGEIO_FFMPEG_EXE', None)
    if exe:
        return exe
    exe = _get_ffmpeg_exe()
    if exe:
        return exe
    raise RuntimeError('No ffmpeg exe could be found. Install ffmpeg on your system, or set the IMAGEIO_FFMPEG_EXE environment variable.')