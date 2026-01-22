from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def compile_setLink(self, attdict, content, extra, program):
    dest = attdict['destination']
    colorname = attdict.get('color', 'blue')
    Link = DefDestination(dest)
    program.append(('push',))
    if colorname:
        program.append(('color', colorname))
    program.append(('lineOperation', Link))
    if colorname:
        program.append(('lineOperation', UNDERLINE))
    for e in content:
        self.compileComponent(e, program)
    if colorname:
        program.append(('endLineOperation', UNDERLINE))
    program.append(('endLineOperation', Link))
    program.append(('pop',))