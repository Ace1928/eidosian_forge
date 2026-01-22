import os
import importlib.util
import sys
from distutils.core import Command
from distutils.errors import DistutilsOptionError
Get the list of files that are input to this command, ie. the
        files that get installed as they are named in the build tree.
        The files in this list correspond one-to-one to the output
        filenames returned by 'get_outputs()'.
        