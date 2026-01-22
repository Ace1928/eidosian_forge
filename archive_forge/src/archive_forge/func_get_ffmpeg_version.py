import logging
import os
import subprocess
import sys
from functools import lru_cache
from pkg_resources import resource_filename
from ._definitions import FNAME_PER_PLATFORM, get_platform
def get_ffmpeg_version():
    """
    Get the version of the used ffmpeg executable (as a string).
    """
    exe = get_ffmpeg_exe()
    line = subprocess.check_output([exe, '-version'], **_popen_kwargs()).split(b'\n', 1)[0]
    line = line.decode(errors='ignore').strip()
    version = line.split('version', 1)[-1].lstrip().split(' ', 1)[0].strip()
    return version