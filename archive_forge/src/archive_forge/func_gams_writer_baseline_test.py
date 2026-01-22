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
@parameterized.parameterized.expand(input=validlist)
def gams_writer_baseline_test(self, name, targetdir):
    baseline = join(currdir, name + '.pyomo.gms')
    testFile = TempfileManager.create_tempfile(suffix=name + '.test.gms')
    cmd = ['--output=' + testFile, join(targetdir, name + '_testCase.py')]
    if os.path.exists(join(targetdir, name + '.dat')):
        cmd.append(join(targetdir, name + '.dat'))
    self.pyomo(cmd)
    try:
        self.assertTrue(cmp(testFile, baseline))
    except:
        with open(baseline, 'r') as f1, open(testFile, 'r') as f2:
            f1_contents = list(filter(None, f1.read().split()))
            f2_contents = list(filter(None, f2.read().split()))
            self.assertEqual(f1_contents, f2_contents, '\n\nbaseline: %s\ntestFile: %s\n' % (baseline, testFile))