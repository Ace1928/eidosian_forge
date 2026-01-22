import os
import re
import shutil
import sys
import tempfile
import zipfile
from glob import glob
from os.path import abspath
from os.path import join as pjoin
from subprocess import PIPE, Popen
import os
import sys
import {mod_name}
def get_sdist_finder(mod_name):
    """Return function finding sdist source directory for `mod_name`"""

    def pf(pth):
        pkg_dirs = glob(pjoin(pth, mod_name + '-*'))
        if len(pkg_dirs) != 1:
            raise OSError('There must be one and only one package dir')
        return pkg_dirs[0]
    return pf