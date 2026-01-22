from math import sqrt
from .gui import *
def make_faint(self):
    for line in self.lines:
        self.canvas.itemconfig(line, fill='gray', width=1)
    self.style = 'faint'