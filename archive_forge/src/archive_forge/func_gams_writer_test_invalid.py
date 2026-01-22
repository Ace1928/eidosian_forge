import re
import glob
import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import attempt_import
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
import pyomo.scripting.pyomo_main as main
@parameterized.parameterized.expand(input=invalidlist)
def gams_writer_test_invalid(self, name, targetdir):
    with self.assertRaisesRegex(RuntimeError, 'GAMS files cannot represent the unary function'):
        testFile = TempfileManager.create_tempfile(suffix=name + '.test.gms')
        cmd = ['--output=' + testFile, join(targetdir, name + '_testCase.py')]
        if os.path.exists(join(targetdir, name + '.dat')):
            cmd.append(join(targetdir, name + '.dat'))
        self.pyomo(cmd)