import re, textwrap, os
from os import sys, path
from distutils.errors import DistutilsError
def opt(self):
    if not self._opt:
        self._opt = self.nopt()
    return self._opt