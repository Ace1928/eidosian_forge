import sys, os
import types
from . import model
from .error import VerificationError
def patch_extension_kwds(self, kwds):
    kwds.setdefault('export_symbols', self.export_symbols)