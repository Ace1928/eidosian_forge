from io import BytesIO
from fontTools import cffLib
from . import DefaultTable
def haveGlyphNames(self):
    if hasattr(self.cff[self.cff.fontNames[0]], 'ROS'):
        return False
    else:
        return True