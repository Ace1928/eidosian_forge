import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import types
def clear_directory(self, path):
    for fn in os.listdir(path):
        fn = os.path.join(path, fn)
        if os.path.islink(fn) or os.path.isfile(fn):
            os.remove(fn)
        elif os.path.isdir(fn):
            shutil.rmtree(fn)