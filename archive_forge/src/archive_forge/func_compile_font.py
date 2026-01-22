from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def compile_font(self, attdict, content, extra, program):
    program.append(('push',))
    if 'face' in attdict:
        face = attdict['face']
        from reportlab.lib.fonts import tt2ps
        try:
            font = tt2ps(face, self.bold, self.italic)
        except:
            font = face
        program.append(('face', font))
    if 'color' in attdict:
        colorname = attdict['color']
        program.append(('color', colorname))
    if 'size' in attdict:
        size = attdict['size']
        program.append(('size', size))
    for e in content:
        self.compileComponent(e, program)
    program.append(('pop',))