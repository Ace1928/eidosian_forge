import os
import sys
import traceback
import bpython
from bpython.args import version_banner, copyright_banner
from .debugger import BPdb
from optparse import OptionParser
from pdb import Restart
def pm():
    post_mortem(getattr(sys, 'last_traceback', None))