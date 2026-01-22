from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def pushTextState(self):
    state = []
    for var in self.TEXT_STATE_VARIABLES:
        val = getattr(self, var)
        state.append(val)
    self.textStateStack = self.textStateStack + [state]
    return state