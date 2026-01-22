from reportlab.lib.units import cm
from reportlab.lib.utils import commasplit, escapeOnce, encode_label, decode_label, strTypes, asUnicode, asNative
from reportlab.lib.styles import ParagraphStyle, _baseFontName
from reportlab.lib import sequencer as rl_sequencer
from reportlab.platypus.paragraph import Paragraph
from reportlab.platypus.doctemplate import IndexingFlowable
from reportlab.platypus.tables import TableStyle, Table
from reportlab.platypus.flowables import Spacer
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
import unicodedata
from ast import literal_eval
def drawPageNumbers(canvas, style, pages, availWidth, availHeight, dot=' . ', formatter=None):
    """
    Draws pagestr on the canvas using the given style.
    If dot is None, pagestr is drawn at the current position in the canvas.
    If dot is a string, pagestr is drawn right-aligned. If the string is not empty,
    the gap is filled with it.
    """
    pagestr = ', '.join([str(p) for p, _ in pages])
    x, y = (canvas._curr_tx_info['cur_x'], canvas._curr_tx_info['cur_y'])
    fontSize = style.fontSize
    pagestrw = stringWidth(pagestr, style.fontName, fontSize)
    freeWidth = availWidth - x
    while pagestrw > freeWidth and fontSize >= 1.0:
        fontSize = 0.9 * fontSize
        pagestrw = stringWidth(pagestr, style.fontName, fontSize)
    if isinstance(dot, strTypes):
        if dot:
            dotw = stringWidth(dot, style.fontName, fontSize)
            dotsn = int((availWidth - x - pagestrw) / dotw)
        else:
            dotsn = dotw = 0
        text = '%s%s' % (dotsn * dot, pagestr)
        newx = availWidth - dotsn * dotw - pagestrw
        pagex = availWidth - pagestrw
    elif dot is None:
        text = ',  ' + pagestr
        newx = x
        pagex = newx
    else:
        raise TypeError('Argument dot should either be None or an instance of basestring.')
    tx = canvas.beginText(newx, y)
    tx.setFont(style.fontName, fontSize)
    tx.setFillColor(style.textColor)
    tx.textLine(text)
    canvas.drawText(tx)
    commaw = stringWidth(', ', style.fontName, fontSize)
    for p, key in pages:
        if not key:
            continue
        w = stringWidth(str(p), style.fontName, fontSize)
        canvas.linkRect('', key, (pagex, y, pagex + w, y + style.leading), relative=1)
        pagex += w + commaw