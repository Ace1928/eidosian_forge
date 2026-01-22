from copy import copy
import numpy as np
import proglog
from tqdm import tqdm
from moviepy.decorators import (apply_to_audio, apply_to_mask,
@outplace
def set_fps(self, fps):
    """ Returns a copy of the clip with a new default fps for functions like
        write_videofile, iterframe, etc. """
    self.fps = fps