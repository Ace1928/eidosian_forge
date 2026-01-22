import codecs
from datetime import timezone
from datetime import datetime
from enum import Enum
from functools import total_ordering
from io import BytesIO
import itertools
import logging
import math
import os
import string
import struct
import sys
import time
import types
import warnings
import zlib
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import _api, _text_helpers, _type1font, cbook, dviread
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.figure import Figure
from matplotlib.font_manager import get_font, fontManager as _fontManager
from matplotlib._afm import AFM
from matplotlib.ft2font import (FIXED_WIDTH, ITALIC, LOAD_NO_SCALE,
from matplotlib.transforms import Affine2D, BboxBase
from matplotlib.path import Path
from matplotlib.dates import UTC
from matplotlib import _path
from . import _backend_pdf_ps
def clip_cmd(self, cliprect, clippath):
    """Set clip rectangle. Calls `.pop()` and `.push()`."""
    cmds = []
    while (self._cliprect, self._clippath) != (cliprect, clippath) and self.parent is not None:
        cmds.extend(self.pop())
    if (self._cliprect, self._clippath) != (cliprect, clippath) or self.parent is None:
        cmds.extend(self.push())
        if self._cliprect != cliprect:
            cmds.extend([cliprect, Op.rectangle, Op.clip, Op.endpath])
        if self._clippath != clippath:
            path, affine = clippath.get_transformed_path_and_affine()
            cmds.extend(PdfFile.pathOperations(path, affine, simplify=False) + [Op.clip, Op.endpath])
    return cmds