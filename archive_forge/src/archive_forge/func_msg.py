import sys
import os
import getopt
from pyparsing import *
def msg(txt):
    """Send message to stdout."""
    sys.stdout.write(txt)
    sys.stdout.flush()