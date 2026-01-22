import os
from glob import glob
import re
from collections.abc import Sequence
from copy import copy
import numpy as np
from PIL import Image
from tifffile import TiffFile
@property
def conserve_memory(self):
    return self._conserve_memory