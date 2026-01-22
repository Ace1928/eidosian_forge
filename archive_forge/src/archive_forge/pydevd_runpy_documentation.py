importers when locating support scripts as well as when importing modules.
import sys
import importlib.machinery  # importlib first so we can test #15386 via -m
import importlib.util
import io
import types
import os
Execute code located at the specified filesystem location

       Returns the resulting top level namespace dictionary

       The file path may refer directly to a Python script (i.e.
       one that could be directly executed with execfile) or else
       it may refer to a zipfile or directory containing a top
       level __main__.py script.
    