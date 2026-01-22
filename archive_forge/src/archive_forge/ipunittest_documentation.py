import re
import sys
import unittest
from doctest import DocTestFinder, DocTestRunner, TestResults
from IPython.terminal.interactiveshell import InteractiveShell
Use as a decorator: doctest a function's docstring as a unittest.
        
        This version runs normal doctests, but the idea is to make it later run
        ipython syntax instead.