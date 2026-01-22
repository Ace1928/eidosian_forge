import os
import sys
import py
import tempfile
def readouterr(self):
    """ return snapshot value of stdout/stderr capturings. """
    out = err = ''
    if self.out:
        out = self.out.getvalue()
        self.out.truncate(0)
        self.out.seek(0)
    if self.err:
        err = self.err.getvalue()
        self.err.truncate(0)
        self.err.seek(0)
    return (out, err)