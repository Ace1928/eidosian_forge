import subprocess as sp
from ..utils import *
from .. import _HAS_MEDIAINFO
from .. import _MEDIAINFO_APPLICATION
get metadata by using mediainfo

    Checks the output of mediainfo on the desired video
    file. Data is then parsed into a dictionary and
    checked for video data. If no such video data exists,
    an empty dictionary is returned.

    Parameters
    ----------
    filename : string
        Path to the video file

    Returns
    -------
    mediaDict : dict
       Dictionary containing all header-based information 
       about the passed-in source video.

    