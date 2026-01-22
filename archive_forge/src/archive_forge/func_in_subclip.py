import re
import numpy as np
from moviepy.tools import cvsecs
from moviepy.video.VideoClip import TextClip, VideoClip
def in_subclip(self, t_start=None, t_end=None):
    """ Returns a sequence of [(t1,t2), txt] covering all the given subclip
        from t_start to t_end. The first and last times will be cropped so as
        to be exactly t_start and t_end if possible. """

    def is_in_subclip(t1, t2):
        try:
            return t_start <= t1 < t_end or t_start < t2 <= t_end
        except:
            return False

    def try_cropping(t1, t2):
        try:
            return (max(t1, t_start), min(t2, t_end))
        except:
            return (t1, t2)
    return [(try_cropping(t1, t2), txt) for (t1, t2), txt in self.subtitles if is_in_subclip(t1, t2)]