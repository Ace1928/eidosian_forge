import os
import sys
from os.path import dirname, isdir, isfile
from os.path import join as pjoin
from os.path import pathsep, realpath
from subprocess import PIPE, Popen
def local_module_dir(module_name):
    """Get local module directory if running in development dir, else None"""
    mod = __import__(module_name)
    containing_path = dirname(dirname(realpath(mod.__file__)))
    if containing_path == realpath(os.getcwd()):
        return containing_path
    return None