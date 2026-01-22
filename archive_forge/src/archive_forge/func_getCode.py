from reportlab.pdfgen import pdfgeom
from reportlab.lib.rl_accel import fp_str
def getCode(self):
    """pack onto one line; used internally"""
    return ' '.join(self._code)