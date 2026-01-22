import breezy.branch
from breezy import branch as _mod_branch
from breezy import check, controldir, errors, gpg, osutils
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import transport, ui, urlutils, workingtree
from breezy.bzr import bzrdir as _mod_bzrdir
from breezy.bzr.remote import (RemoteBzrDir, RemoteBzrDirFormat,
from breezy.tests import (ChrootedTestCase, TestNotApplicable, TestSkipped,
from breezy.tests.per_controldir import TestCaseWithControlDir
from breezy.transport.local import LocalTransport
from breezy.ui import CannedInputUIFactory
def test_backup_copies_existing(self):
    tree = self.make_branch_and_tree('test')
    self.build_tree(['test/a'])
    tree.add(['a'])
    tree.commit('some data to be copied.')
    old_url, new_url = tree.controldir.backup_bzrdir()
    old_path = urlutils.local_path_from_url(old_url)
    new_path = urlutils.local_path_from_url(new_url)
    self.assertPathExists(old_path)
    self.assertPathExists(new_path)
    for ((dir_relpath1, _), entries1), ((dir_relpath2, _), entries2) in zip(osutils.walkdirs(old_path), osutils.walkdirs(new_path)):
        self.assertEqual(dir_relpath1, dir_relpath2)
        for f1, f2 in zip(entries1, entries2):
            self.assertEqual(f1[0], f2[0])
            self.assertEqual(f1[2], f2[2])
            if f1[2] == 'file':
                with open(f1[4], 'rb') as a, open(f2[4], 'rb') as b:
                    osutils.compare_files(a, b)