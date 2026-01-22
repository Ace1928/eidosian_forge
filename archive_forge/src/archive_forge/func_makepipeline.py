from a disk file or from an open file, and similar for its output.
import re
import os
import tempfile
import warnings
from shlex import quote
def makepipeline(self, infile, outfile):
    cmd = makepipeline(infile, self.steps, outfile)
    if self.debugging:
        print(cmd)
        cmd = 'set -x; ' + cmd
    return cmd