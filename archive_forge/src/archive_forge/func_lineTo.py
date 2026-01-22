from reportlab.pdfgen import pdfgeom
from reportlab.lib.rl_accel import fp_str
def lineTo(self, x, y):
    self._code_append('%s l' % fp_str(x, y))