from itertools import zip_longest
import json
import re
import glob
import subprocess
import os
from os.path import join
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import attempt_import
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
import pyomo.scripting.pyomo_main as main
@parameterized.parameterized.expand(input=names)
def nlwriter_asl_test(self, name):
    testFile = TempfileManager.create_tempfile(suffix=name + '.test.nl')
    testFile_row = testFile[:-2] + 'row'
    TempfileManager.add_tempfile(testFile_row, exists=False)
    testFile_col = testFile[:-2] + 'col'
    TempfileManager.add_tempfile(testFile_col, exists=False)
    cmd = ['--output=' + testFile, '--file-determinism=2', '--symbolic-solver-labels', join(currdir, name + '_testCase.py')]
    if os.path.exists(join(currdir, name + '.dat')):
        cmd.append(join(currdir, name + '.dat'))
    self.pyomo(cmd)
    testFile_json = testFile[:-2] + 'json'
    TempfileManager.add_tempfile(testFile_json, exists=False)
    p = subprocess.run(['gjh_asl_json', testFile, 'rows=' + testFile_row, 'cols=' + testFile_col, 'json=' + testFile_json], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    self.assertTrue(p.returncode == 0, msg=p.stdout)
    baseFile = join(currdir, name + '.ampl.nl')
    amplFile = TempfileManager.create_tempfile(suffix=name + '.ampl.json')
    p = subprocess.run(['gjh_asl_json', baseFile, 'rows=' + baseFile[:-2] + 'row', 'cols=' + baseFile[:-2] + 'col', 'json=' + amplFile], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    self.assertTrue(p.returncode == 0, msg=p.stdout)
    with open(testFile_json, 'r') as f1, open(amplFile, 'r') as f2:
        self.assertStructuredAlmostEqual(json.load(f1), json.load(f2), abstol=1e-08)