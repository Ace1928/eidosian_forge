import sys, os, binascii, shutil, io
from . import __version_verifier_modules__
from . import ffiplatform
from .error import VerificationError
def generates_python_module(self):
    return self._vengine._gen_python_module