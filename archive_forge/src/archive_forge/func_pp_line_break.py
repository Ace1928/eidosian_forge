import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_line_break(self, f, indent):
    self.pos = indent
    self.ribbon_pos = 0
    self.line = self.line + 1
    if self.line < self.max_lines:
        self.out.write(u('\n'))
        for i in range(indent):
            self.out.write(u(' '))
    else:
        self.out.write(u('\n...'))
        raise StopPPException()