from __future__ import print_function
from reportlab.graphics.barcode.common import Barcode
from reportlab.lib.utils import asNative
@pitch.setter
def pitch(self, value):
    n, x = self.dimensions['pitch']
    self.__dict__['_pitch'] = 72 * min(max(value / 72.0, n), x)