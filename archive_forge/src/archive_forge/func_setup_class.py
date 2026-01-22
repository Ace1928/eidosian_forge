import re, textwrap, os
from os import sys, path
from distutils.errors import DistutilsError
def setup_class(self):
    FakeCCompilerOpt.conf_nocache = True
    self._opt = None