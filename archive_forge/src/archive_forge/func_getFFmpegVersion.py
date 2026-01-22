from .utils import check_output, where
import os
import warnings
import numpy as np
def getFFmpegVersion():
    """ Returns the version of FFmpeg that is currently being used
    """
    if _FFMPEG_MAJOR_VERSION[0] == 'N':
        return '%s' % (_FFMPEG_MAJOR_VERSION,)
    else:
        return '%s.%s.%s' % (_FFMPEG_MAJOR_VERSION, _FFMPEG_MINOR_VERSION, _FFMPEG_PATCH_VERSION)