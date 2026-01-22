import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def generate_hg_version_py():
    if not os.path.isfile(target):
        version = str(revision)
        self.info('Creating %s (version=%r)' % (target, version))
        with open(target, 'w') as f:
            f.write('version = %r\n' % version)

    def rm_file(f=target, p=self.info):
        if delete:
            try:
                os.remove(f)
                p('removed ' + f)
            except OSError:
                pass
            try:
                os.remove(f + 'c')
                p('removed ' + f + 'c')
            except OSError:
                pass
    atexit.register(rm_file)
    return target