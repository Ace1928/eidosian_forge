from reportlab.graphics.shapes import *
from reportlab.lib.validators import DerivedValue
from reportlab import rl_config
from . transform import mmult, inverse
def getCTM(self):
    """returns the current transformation matrix at this point"""
    return self._combined[-1]['ctm']