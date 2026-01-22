from reportlab.pdfbase.pdfmetrics import getFont, unicode2T1
from reportlab.lib.utils import open_and_read, isBytes, rl_exec
from .shapes import _baseGFontName, _PATH_OP_ARG_COUNT, _PATH_OP_NAMES, definePath
from sys import exc_info
def cubic_to(a, b, c, ctx):
    P_append(('curveTo', xpt(a.x), ypt(a.y), xpt(b.x), ypt(b.y), xpt(c.x), ypt(c.y)))