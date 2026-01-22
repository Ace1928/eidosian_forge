import os
import subprocess as sp
import numpy as np
import proglog
from moviepy.compat import DEVNULL
from moviepy.config import get_setting
from moviepy.decorators import requires_duration, use_clip_fps_by_default
from moviepy.tools import subprocess_call

    Writes the gif with the Python library ImageIO (calls FreeImage).

    Parameters
    -----------
    opt

    