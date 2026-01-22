import codecs
import datetime
import functools
from io import BytesIO
import logging
import math
import os
import pathlib
import shutil
import subprocess
from tempfile import TemporaryDirectory
import weakref
from PIL import Image
import matplotlib as mpl
from matplotlib import _api, cbook, font_manager as fm
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.backends.backend_pdf import (
from matplotlib.path import Path
from matplotlib.figure import Figure
from matplotlib._pylab_helpers import Gcf
def make_pdf_to_png_converter():
    """Return a function that converts a pdf file to a png file."""
    try:
        mpl._get_executable_info('pdftocairo')
    except mpl.ExecutableNotFoundError:
        pass
    else:
        return lambda pdffile, pngfile, dpi: subprocess.check_output(['pdftocairo', '-singlefile', '-transp', '-png', '-r', '%d' % dpi, pdffile, os.path.splitext(pngfile)[0]], stderr=subprocess.STDOUT)
    try:
        gs_info = mpl._get_executable_info('gs')
    except mpl.ExecutableNotFoundError:
        pass
    else:
        return lambda pdffile, pngfile, dpi: subprocess.check_output([gs_info.executable, '-dQUIET', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-dNOPROMPT', '-dUseCIEColor', '-dTextAlphaBits=4', '-dGraphicsAlphaBits=4', '-dDOINTERPOLATE', '-sDEVICE=pngalpha', '-sOutputFile=%s' % pngfile, '-r%d' % dpi, pdffile], stderr=subprocess.STDOUT)
    raise RuntimeError('No suitable pdf to png renderer found.')