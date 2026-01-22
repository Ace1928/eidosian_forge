import os
import re
import shlex
import sys
import pygments
from pathlib import Path
from IPython.utils.text import marquee
from IPython.utils import openpy
from IPython.utils import py3compat
def fload(self):
    """Load file object."""
    if hasattr(self, 'fobj') and self.fobj is not None:
        self.fobj.close()
    if hasattr(self.src, 'read'):
        self.fobj = self.src
    else:
        self.fobj = openpy.open(self.fname)