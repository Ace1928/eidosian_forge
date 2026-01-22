from io import BytesIO
from reportlab.graphics.shapes import *
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab import rl_config
from reportlab.graphics.renderbase import Renderer, getStateDelta, renderScaledDrawing, STATE_DEFAULTS
from reportlab.platypus import Flowable
def drawToFile(d, fn, msg='', showBoundary=rl_config._unset_, autoSize=1, canvasKwds={}):
    """Makes a one-page PDF with just the drawing.

    If autoSize=1, the PDF will be the same size as
    the drawing; if 0, it will place the drawing on
    an A4 page with a title above it - possibly overflowing
    if too big."""
    d = renderScaledDrawing(d)
    for x in ('Name', 'Size'):
        a = 'initialFont' + x
        canvasKwds[a] = getattr(d, a, canvasKwds.pop(a, STATE_DEFAULTS['font' + x]))
    c = Canvas(fn, **canvasKwds)
    if msg:
        c.setFont(rl_config.defaultGraphicsFontName, 36)
        c.drawString(80, 750, msg)
    c.setTitle(msg)
    if autoSize:
        c.setPageSize((d.width, d.height))
        draw(d, c, 0, 0, showBoundary=showBoundary)
    else:
        c.setFont(rl_config.defaultGraphicsFontName, 12)
        y = 740
        i = 1
        y = y - d.height
        draw(d, c, 80, y, showBoundary=showBoundary)
    c.showPage()
    c.save()
    if sys.platform == 'mac' and (not hasattr(fn, 'write')):
        try:
            import macfs, macostools
            macfs.FSSpec(fn).SetCreatorType('CARO', 'PDF ')
            macostools.touched(fn)
        except:
            pass