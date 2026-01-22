import os
import re
import sys
from collections import deque
from io import StringIO
def push_source(self, newstream, newfile=None):
    """Push an input source onto the lexer's input source stack."""
    if isinstance(newstream, str):
        newstream = StringIO(newstream)
    self.filestack.appendleft((self.infile, self.instream, self.lineno))
    self.infile = newfile
    self.instream = newstream
    self.lineno = 1
    if self.debug:
        if newfile is not None:
            print('shlex: pushing to file %s' % (self.infile,))
        else:
            print('shlex: pushing to stream %s' % (self.instream,))