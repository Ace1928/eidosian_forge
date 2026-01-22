import os
import sys
from breezy import osutils, tests, urlutils
from breezy.tests import EncodingAdapter
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_ignored(self):
    fname = self.info['filename'] + '1.txt'
    self.build_tree_contents([(fname, b'ignored\n')])
    self.run_bzr(['ignore', fname])
    txt = self.run_bzr_decode(['ignored'])
    self.assertEqual(txt, '%-50s %s\n' % (fname, fname))
    txt = self.run_bzr_decode(['ignored'], encoding='ascii')
    fname = fname.encode('ascii', 'replace').decode('ascii')
    self.assertEqual(txt, '%-50s %s\n' % (fname, fname))