import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def my_link_shared_object(self, *args, **kwds):
    if '-bundle' in self.linker_so:
        self.linker_so = list(self.linker_so)
        i = self.linker_so.index('-bundle')
        self.linker_so[i] = '-dynamiclib'
    return old_link_shared_object(self, *args, **kwds)