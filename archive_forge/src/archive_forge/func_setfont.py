from __future__ import annotations
import sys
from . import EpsImagePlugin
def setfont(self, font, size):
    """
        Selects which font to use.

        :param font: A PostScript font name
        :param size: Size in points.
        """
    font = bytes(font, 'UTF-8')
    if font not in self.isofont:
        self.fp.write(b'/PSDraw-%s ISOLatin1Encoding /%s E\n' % (font, font))
        self.isofont[font] = 1
    self.fp.write(b'/F0 %d /PSDraw-%s F\n' % (size, font))