import formatter
import string
from types import *
import htmllib
import piddle
def new_spacing(self, spacing):
    self.send_line_break()
    t = 'new_spacing(%s)' % repr(spacing)
    self.OutputLine(t, 1)