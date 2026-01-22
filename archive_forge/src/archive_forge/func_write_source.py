import sys, os, binascii, shutil, io
from . import __version_verifier_modules__
from . import ffiplatform
from .error import VerificationError
def write_source(self, file=None):
    """Write the C source code.  It is produced in 'self.sourcefilename',
        which can be tweaked beforehand."""
    with self.ffi._lock:
        if self._has_source and file is None:
            raise VerificationError('source code already written')
        self._write_source(file)