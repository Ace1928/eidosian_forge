import os
import numpy as np
from imageio import imread
from ..VideoClip import VideoClip
def mask_make_frame(t):
    index = find_image_index(t)
    return 1.0 * self.sequence[index][:, :, 3] / 255