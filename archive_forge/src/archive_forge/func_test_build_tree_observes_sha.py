import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
def test_build_tree_observes_sha(self):
    source = self.make_branch_and_tree('source')
    self.build_tree(['source/file1', 'source/dir/', 'source/dir/file2'])
    source.add(['file1', 'dir', 'dir/file2'], ids=[b'file1-id', b'dir-id', b'file2-id'])
    source.commit('new files')
    target = self.make_branch_and_tree('target')
    target.lock_write()
    self.addCleanup(target.unlock)
    state = target.current_dirstate()
    state._cutoff_time = time.time() + 60
    build_tree(source.basis_tree(), target)
    entry1_sha = osutils.sha_file_by_name('source/file1')
    entry2_sha = osutils.sha_file_by_name('source/dir/file2')
    entry1 = state._get_entry(0, path_utf8=b'file1')
    self.assertEqual(entry1_sha, entry1[1][0][1])
    self.assertEqual(25, entry1[1][0][2])
    entry1_state = entry1[1][0]
    entry2 = state._get_entry(0, path_utf8=b'dir/file2')
    self.assertEqual(entry2_sha, entry2[1][0][1])
    self.assertEqual(29, entry2[1][0][2])
    entry2_state = entry2[1][0]
    self.assertEqual(entry1_sha, target.get_file_sha1('file1'))
    self.assertEqual(entry2_sha, target.get_file_sha1('dir/file2'))
    self.assertEqual(entry1_state, entry1[1][0])
    self.assertEqual(entry2_state, entry2[1][0])