from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def compile_para(self, attdict, content, extra, program, stylename='para.defaultStyle'):
    if attdict is None:
        attdict = {}
    context = self.context
    stylename = attdict.get('style', stylename)
    style = context[stylename]
    newstyle = SimpleStyle(name='rml2pdf internal embedded style', parent=style)
    newstyle.addAttributes(attdict)
    bulletText = attdict.get('bulletText', None)
    mystyle = self.style1
    thepara = Para(newstyle, content, context=context, bulletText=bulletText)
    mybaseindent = self.baseindent
    self.baseindent = thepara.baseindent = mystyle.leftIndent + self.baseindent
    thepara.linearize(program=program)
    program.append(('nextLine', 0))
    self.baseindent = mybaseindent