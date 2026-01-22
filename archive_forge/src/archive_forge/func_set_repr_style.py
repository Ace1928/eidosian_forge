import py
import sys
from inspect import CO_VARARGS, CO_VARKEYWORDS, isclass
import traceback
def set_repr_style(self, mode):
    assert mode in ('short', 'long')
    self._repr_style = mode