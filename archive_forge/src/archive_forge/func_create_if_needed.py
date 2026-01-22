import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import types
def create_if_needed(d):
    if not os.path.exists(d):
        os.makedirs(d)
    elif os.path.islink(d) or os.path.isfile(d):
        raise ValueError('Unable to create directory %r' % d)