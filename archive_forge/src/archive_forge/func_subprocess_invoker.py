import os
import shutil
import subprocess
import sys
import tempfile
from .lazy_import import lazy_import
from breezy import (
def subprocess_invoker(executable, args, cleanup):
    retcode = subprocess.call([executable] + args)
    cleanup(retcode)
    return retcode