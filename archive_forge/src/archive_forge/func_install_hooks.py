from __future__ import absolute_import, division, print_function
import sys
import logging
import contextlib
import copy
import os
from future.utils import PY2, PY3
def install_hooks():
    """
    This function installs the future.standard_library import hook into
    sys.meta_path.
    """
    if PY3:
        return
    install_aliases()
    flog.debug('sys.meta_path was: {0}'.format(sys.meta_path))
    flog.debug('Installing hooks ...')
    newhook = RenameImport(RENAMES)
    if not detect_hooks():
        sys.meta_path.append(newhook)
    flog.debug('sys.meta_path is now: {0}'.format(sys.meta_path))