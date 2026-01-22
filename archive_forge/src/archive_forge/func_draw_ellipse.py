import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def draw_ellipse(self, drawop, style=None):
    op, x, y, w, h = drawop
    s = ''
    if op == 'E':
        if self.opacity is not None:
            cmd = 'filldraw [opacity=%s]' % self.opacity
        else:
            cmd = 'filldraw'
    else:
        cmd = 'draw'
    if style:
        stylestr = ' [%s]' % style
    else:
        stylestr = ''
    s += '  \\%s%s (%sbp,%sbp) ellipse (%sbp and %sbp);\n' % (cmd, stylestr, smart_float(x), smart_float(y), smart_float(w), smart_float(h))
    return s