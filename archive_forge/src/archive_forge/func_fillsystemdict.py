from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes, tostr
from fontTools.misc import eexec
from .psOperators import (
import re
from collections.abc import Callable
from string import whitespace
import logging
def fillsystemdict(self):
    systemdict = self.dictstack[0]
    systemdict['['] = systemdict['mark'] = self.mark = ps_mark()
    systemdict[']'] = ps_operator(']', self.do_makearray)
    systemdict['true'] = ps_boolean(1)
    systemdict['false'] = ps_boolean(0)
    systemdict['StandardEncoding'] = ps_array(ps_StandardEncoding)
    systemdict['FontDirectory'] = ps_dict({})
    self.suckoperators(systemdict, self.__class__)