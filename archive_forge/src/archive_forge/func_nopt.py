import re, textwrap, os
from os import sys, path
from distutils.errors import DistutilsError
def nopt(self, *args, **kwargs):
    FakeCCompilerOpt.fake_info = (self.arch, self.cc, '')
    return FakeCCompilerOpt(*args, **kwargs)