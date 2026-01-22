import formatter
import string
from types import *
import htmllib
import piddle
def send_line_break(self):
    if self.lineHeight:
        self.y = self.y + self.lineHeight
        self.oldLineHeight = self.lineHeight
        self.lineHeight = 0
    self.x = self.indent
    self.atbreak = 0
    if TRACE:
        input('lb')