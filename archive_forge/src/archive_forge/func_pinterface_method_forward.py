import sys
import ctypes
from pyglet.util import debug_print
def pinterface_method_forward(self, *args, _m=m, _i=i):
    assert _debug_com(f'Calling COM {_i} of {target.__name__} ({_m}) through pointer: ({', '.join(map(repr, (self, *args)))})')
    return _m(self, *args)