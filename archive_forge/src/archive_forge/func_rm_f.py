import optparse
import os
import sys
from subprocess import PIPE, CalledProcessError, Popen
from breezy import osutils
from breezy.tests import ssl_certs
def rm_f(path):
    """rm -f path"""
    try:
        os.unlink(path)
    except:
        pass