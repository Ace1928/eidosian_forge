from reportlab.pdfbase.pdfmetrics import getFont, unicode2T1
from reportlab.lib.utils import open_and_read, isBytes, rl_exec
from .shapes import _baseGFontName, _PATH_OP_ARG_COUNT, _PATH_OP_NAMES, definePath
from sys import exc_info
def move_to(a, ctx):
    if P:
        P_append(('closePath',))
    P_append(('moveTo', xpt(a.x), ypt(a.y)))