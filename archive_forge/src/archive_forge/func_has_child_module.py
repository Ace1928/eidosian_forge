import argparse
import glob
import locale
import os
import sys
from copy import copy
from fnmatch import fnmatch
from importlib.machinery import EXTENSION_SUFFIXES
from os import path
from typing import Any, Generator, List, Optional, Tuple
import sphinx.locale
from sphinx import __display_version__, package_dir
from sphinx.cmd.quickstart import EXTENSIONS
from sphinx.locale import __
from sphinx.util.osutil import FileAvoidWrite, ensuredir
from sphinx.util.template import ReSTRenderer
def has_child_module(rootpath: str, excludes: List[str], opts: Any) -> bool:
    """Check the given directory contains child module/s (at least one)."""
    for _root, _subs, files in walk(rootpath, excludes, opts):
        if files:
            return True
    return False