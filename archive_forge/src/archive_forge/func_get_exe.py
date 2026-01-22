import re
import sys
import time
import logging
import platform
import threading
import subprocess as sp
import imageio_ffmpeg
import numpy as np
from ..core import Format, image_as_uint
def get_exe():
    """Wrapper for imageio_ffmpeg.get_ffmpeg_exe()"""
    return imageio_ffmpeg.get_ffmpeg_exe()