from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes, tostr
from fontTools.misc import eexec
from .psOperators import (
import re
from collections.abc import Callable
from string import whitespace
import logging
def do_makearray(self):
    array = []
    while 1:
        topobject = self.pop()
        if topobject == self.mark:
            break
        array.append(topobject)
    array.reverse()
    self.push(ps_array(array))