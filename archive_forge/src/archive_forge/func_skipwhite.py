from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes, tostr
from fontTools.misc import eexec
from .psOperators import (
import re
from collections.abc import Callable
from string import whitespace
import logging
def skipwhite(self, whitematch=skipwhiteRE.match):
    _, nextpos = whitematch(self.buf, self.pos).span()
    self.pos = nextpos