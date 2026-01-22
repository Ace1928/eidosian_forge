import glob
import importlib
import os
import subprocess
import sys
from itertools import filterfalse
from os.path import join
import pyomo.common.dependencies as dependencies
from pyomo.common.fileutils import PYOMO_ROOT_DIR
import pyomo.common.unittest as unittest
import_test = """
import pyomo.common.dependencies
import unittest
def test_for_init_files(self):
    _NMD = set(_NON_MODULE_DIRS)
    fail = []
    module_dir = os.path.join(PYOMO_ROOT_DIR, 'pyomo')
    for path, subdirs, files in os.walk(module_dir):
        assert path.startswith(module_dir)
        relpath = path[1 + len(module_dir):]
        try:
            subdirs.remove('__pycache__')
        except ValueError:
            pass
        if '__init__.py' in files:
            continue
        if relpath in _NMD:
            _NMD.remove(relpath)
            subdirs[:] = []
            continue
        fail.append(relpath)
    if fail:
        self.fail('Directories are missing __init__.py files:\n\t' + '\n\t'.join(sorted(fail)))
    if _NMD:
        self.fail('_NON_MODULE_DIRS contains entries not found in package or unexpectedly contain a __init__.py file:\n\t' + '\n\t'.join(sorted(_NMD)))