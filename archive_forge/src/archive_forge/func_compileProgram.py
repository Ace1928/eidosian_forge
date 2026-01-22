from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def compileProgram(self, parsedText, program=None):
    style = self.style1
    if program is None:
        program = []
    a = program.append
    fn = style.fontName
    a(('face', fn))
    from reportlab.lib.fonts import ps2tt
    self.face, self.bold, self.italic = ps2tt(fn)
    a(('size', style.fontSize))
    self.size = style.fontSize
    a(('align', style.alignment))
    a(('indent', style.leftIndent))
    if style.firstLineIndent:
        a(('indent', style.firstLineIndent))
    a(('rightIndent', style.rightIndent))
    a(('leading', style.leading))
    if style.textColor:
        a(('color', style.textColor))
    if self.bulletText:
        self.do_bullet(self.bulletText, program)
    self.compileComponent(parsedText, program)
    if style.firstLineIndent:
        count = 0
        for x in program:
            count += 1
            if isinstance(x, str) or hasattr(x, 'width'):
                break
        program.insert(count, ('indent', -style.firstLineIndent))
    return program