import itertools
import re
import glob
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
import pyomo.common
import pyomo.scripting.pyomo_main as main
@parameterized.parameterized.expand(input=names)
def barwriter_baseline_test(self, name):
    baseline = join(currdir, name + '.pyomo.bar')
    output = join(currdir, name + '.test.bar')
    if not os.path.exists(baseline):
        self.skipTest('baseline file (%s) not found' % (baseline,))
    if os.path.exists(datadir + name + '_testCase.py'):
        testDir = datadir
    else:
        testDir = currdir
    testCase = testDir + name + '_testCase.py'
    if os.path.exists(testDir + name + '.dat'):
        self.pyomo(['--output=' + output, testCase, testDir + name + '.dat'])
    else:
        self.pyomo(['--output=' + output, testCase])
    with open(baseline, 'r') as f1, open(output, 'r') as f2:
        f1_contents = list(filter(None, f1.read().split()))
        f2_contents = list(filter(None, f2.read().split()))
        for item1, item2 in itertools.zip_longest(f1_contents, f2_contents):
            try:
                self.assertAlmostEqual(float(item1), float(item2))
            except:
                self.assertEqual(item1, item2, '\n\nbaseline: %s\ntestFile: %s\n' % (baseline, output))
    os.remove(join(currdir, name + '.test.bar'))