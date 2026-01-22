from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes, tostr
from fontTools.misc import eexec
from .psOperators import (
import re
from collections.abc import Callable
from string import whitespace
import logging
def suckoperators(self, systemdict, klass):
    for name in dir(klass):
        attr = getattr(self, name)
        if isinstance(attr, Callable) and name[:3] == 'ps_':
            name = name[3:]
            systemdict[name] = ps_operator(name, attr)
    for baseclass in klass.__bases__:
        self.suckoperators(systemdict, baseclass)