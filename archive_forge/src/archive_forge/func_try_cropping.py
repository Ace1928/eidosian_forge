import re
import numpy as np
from moviepy.tools import cvsecs
from moviepy.video.VideoClip import TextClip, VideoClip
def try_cropping(t1, t2):
    try:
        return (max(t1, t_start), min(t2, t_end))
    except:
        return (t1, t2)