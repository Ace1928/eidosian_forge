from __future__ import absolute_import
import glob
import sys
import os
import subprocess
from .util import py_str
def unzip_archives(ar_list, env):
    for fname in ar_list:
        if not os.path.exists(fname):
            continue
        if fname.endswith('.zip'):
            subprocess.call(args=['unzip', fname], env=env)
        elif fname.find('.tar') != -1:
            subprocess.call(args=['tar', '-xf', fname], env=env)