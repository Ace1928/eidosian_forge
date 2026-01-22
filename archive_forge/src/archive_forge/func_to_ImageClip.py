import os
import subprocess as sp
import tempfile
import warnings
import numpy as np
import proglog
from imageio import imread, imsave
from ..Clip import Clip
from ..compat import DEVNULL, string_types
from ..config import get_setting
from ..decorators import (add_mask_if_none, apply_to_mask,
from ..tools import (deprecated_version_of, extensions_dict, find_extension,
from .io.ffmpeg_writer import ffmpeg_write_video
from .io.gif_writers import (write_gif, write_gif_with_image_io,
from .tools.drawing import blit
@convert_to_seconds(['t'])
def to_ImageClip(self, t=0, with_mask=True, duration=None):
    """
        Returns an ImageClip made out of the clip's frame at time ``t``,
        which can be expressed in seconds (15.35), in (min, sec),
        in (hour, min, sec), or as a string: '01:03:05.35'.
        """
    newclip = ImageClip(self.get_frame(t), ismask=self.ismask, duration=duration)
    if with_mask and self.mask is not None:
        newclip.mask = self.mask.to_ImageClip(t)
    return newclip