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
def newPage(self, width, height):
    self.endStream()
    self.width, self.height = (width, height)
    contentObject = self.reserveObject('page contents')
    annotsObject = self.reserveObject('annotations')
    thePage = {'Type': Name('Page'), 'Parent': self.pagesObject, 'Resources': self.resourceObject, 'MediaBox': [0, 0, 72 * width, 72 * height], 'Contents': contentObject, 'Annots': annotsObject}
    pageObject = self.reserveObject('page')
    self.writeObject(pageObject, thePage)
    self.pageList.append(pageObject)
    self._annotations.append((annotsObject, self.pageAnnotations))
    self.beginStream(contentObject.id, self.reserveObject('length of content stream'))
    self.output(Name('DeviceRGB'), Op.setcolorspace_stroke)
    self.output(Name('DeviceRGB'), Op.setcolorspace_nonstroke)
    self.output(GraphicsContextPdf.joinstyles['round'], Op.setlinejoin)
    self.pageAnnotations = []