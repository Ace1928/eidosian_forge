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
def paths(self, *paths, **kws):
    """Apply glob to paths and prepend local_path if needed.

        Applies glob.glob(...) to each path in the sequence (if needed) and
        pre-pends the local_path if needed. Because this is called on all
        source lists, this allows wildcard characters to be specified in lists
        of sources for extension modules and libraries and scripts and allows
        path-names be relative to the source directory.

        """
    include_non_existing = kws.get('include_non_existing', True)
    return gpaths(paths, local_path=self.local_path, include_non_existing=include_non_existing)