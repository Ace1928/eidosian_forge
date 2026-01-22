import formatter
import string
from types import *
import htmllib
import piddle
def send_hor_rule(self):
    self.send_line_break()
    self.y = self.y + self.oldLineHeight
    border = self.fsizex
    self.pc.drawLine(border, self.y, self.rmargin - border, self.y, piddle.Color(0.0, 0.0, 200 / 255.0))
    self.y = self.y + self.oldLineHeight