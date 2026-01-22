import os
import sys
from breezy import osutils, tests, urlutils
from breezy.tests import EncodingAdapter
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_touching_revisions(self):
    fname = self.info['filename']
    txt = self.run_bzr_decode(['touching-revisions', fname])
    self._check_OSX_can_roundtrip(self.info['filename'])
    self.assertEqual('     3 added {}\n'.format(fname), txt)
    fname2 = self.info['filename'] + '2'
    self.wt.rename_one(fname, fname2)
    self.wt.commit('Renamed {} => {}'.format(fname, fname2))
    txt = self.run_bzr_decode(['touching-revisions', fname2])
    expected_txt = '     3 added %s\n     4 renamed %s => %s\n' % (fname, fname, fname2)
    self.assertEqual(expected_txt, txt)
    self.run_bzr_decode(['touching-revisions', fname2], encoding='ascii', fail=True)