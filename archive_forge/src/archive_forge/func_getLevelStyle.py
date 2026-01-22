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
def getLevelStyle(self, n):
    """Returns the style for level n, generating and caching styles on demand if not present."""
    if not hasattr(self.textStyle, '__iter__'):
        self.textStyle = [self.textStyle]
    try:
        return self.textStyle[n]
    except IndexError:
        self.textStyle = list(self.textStyle)
        prevstyle = self.getLevelStyle(n - 1)
        self.textStyle.append(ParagraphStyle(name='%s-%d-indented' % (prevstyle.name, n), parent=prevstyle, firstLineIndent=prevstyle.firstLineIndent + 0.2 * cm, leftIndent=prevstyle.leftIndent + 0.2 * cm))
        return self.textStyle[n]