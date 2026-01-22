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
def get_config_cmd(self):
    """
        Returns the numpy.distutils config command instance.
        """
    cmd = get_cmd('config')
    cmd.ensure_finalized()
    cmd.dump_source = 0
    cmd.noisy = 0
    old_path = os.environ.get('PATH')
    if old_path:
        path = os.pathsep.join(['.', old_path])
        os.environ['PATH'] = path
    return cmd