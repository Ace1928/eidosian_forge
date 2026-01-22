import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def needs_version(self, ver):
    self._version = max(self._version, ver)