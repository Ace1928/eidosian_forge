from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes, tostr
from fontTools.misc import eexec
from .psOperators import (
import re
from collections.abc import Callable
from string import whitespace
import logging
def handle_object(self, object):
    if not (self.proclevel or object.literal or object.type == 'proceduretype'):
        if object.type != 'operatortype':
            object = self.resolve_name(object.value)
        if object.literal:
            self.push(object)
        elif object.type == 'proceduretype':
            self.call_procedure(object)
        else:
            object.function()
    else:
        self.push(object)