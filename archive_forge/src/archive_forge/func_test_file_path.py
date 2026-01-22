import os
import sys
from breezy import osutils, tests, urlutils
from breezy.tests import EncodingAdapter
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_file_path(self):
    fname = self.info['filename']
    dirname = self.info['directory']
    self.build_tree_contents([('base/',), (osutils.pathjoin('base', '{}/'.format(dirname)),)])
    self.wt.add('base')
    self.wt.add('base/' + dirname)
    path = osutils.pathjoin('base', dirname, fname)
    self._check_OSX_can_roundtrip(self.info['filename'])
    self.wt.rename_one(fname, path)
    self.wt.commit('moving things around')
    txt = self.run_bzr_decode(['file-path', path])
    txt = self.run_bzr_decode(['file-path', path], encoding='ascii')