import re
import numpy as np
from moviepy.tools import cvsecs
from moviepy.video.VideoClip import TextClip, VideoClip
def make_mask_frame(t):
    sub = add_textclip_if_none(t)
    return self.textclips[sub].mask.get_frame(t) if sub else np.array([[0]])