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
@parameterized.expand(modules)
@unittest.pytest.mark.importtest
def test_module_import(self, module):
    module_file = os.path.join(PYOMO_ROOT_DIR, module.replace('.', os.path.sep)) + '.py'
    pyc = importlib.util.cache_from_source(module_file)
    if os.path.isfile(pyc):
        os.remove(pyc)
    test_code = import_test % module
    if _FAST_TEST:
        from pyomo.common.fileutils import import_file
        import warnings
        try:
            _dep_warn = dependencies.SUPPRESS_DEPENDENCY_WARNINGS
            dependencies.SUPPRESS_DEPENDENCY_WARNINGS = True
            with warnings.catch_warnings():
                warnings.resetwarnings()
                warnings.filterwarnings('error')
                import_file(module_file, clear_cache=True)
        except unittest.SkipTest as e:
            print(e)
        finally:
            dependencies.SUPPRESS_DEPENDENCY_WARNINGS = _dep_warn
    else:
        subprocess.run([sys.executable, '-Werror', '-c', test_code], check=True)