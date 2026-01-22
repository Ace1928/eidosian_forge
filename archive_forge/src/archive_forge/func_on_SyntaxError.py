import sys
import logging
from . import compileUi, loadUi
def on_SyntaxError(self, e):
    """ Handle a SyntaxError exception. """
    sys.stderr.write('Error in input file: %s\n' % e)