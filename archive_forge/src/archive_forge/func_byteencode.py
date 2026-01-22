from __future__ import absolute_import
import re
import sys
def byteencode(self):
    if IS_PYTHON3:
        return _bytes(self)
    else:
        return self.decode('ISO-8859-1').encode('ISO-8859-1')