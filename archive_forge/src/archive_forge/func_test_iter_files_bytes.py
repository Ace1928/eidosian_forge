import re
from io import BytesIO
from ... import branch as _mod_branch
from ... import commit, controldir
from ... import delta as _mod_delta
from ... import errors, gpg, info, repository
from ... import revision as _mod_revision
from ... import tests, transport, upgrade, workingtree
from ...bzr import branch as _mod_bzrbranch
from ...bzr import inventory, knitpack_repo, remote
from ...bzr import repository as bzrrepository
from .. import per_repository, test_server
from ..matchers import *
def test_iter_files_bytes(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/file1', b'foo'), ('tree/file2', b'bar')])
    tree.add(['file1', 'file2'])
    if not tree.supports_file_ids:
        raise tests.TestNotApplicable('tree does not support file ids')
    file1_id = tree.path2id('file1')
    file2_id = tree.path2id('file2')
    rev1 = tree.commit('rev1')
    self.build_tree_contents([('tree/file1', b'baz')])
    rev2 = tree.commit('rev2')
    repository = tree.branch.repository
    repository.lock_read()
    self.addCleanup(repository.unlock)
    extracted = {i: b''.join(b) for i, b in repository.iter_files_bytes([(file1_id, rev1, 'file1-old'), (file1_id, rev2, 'file1-new'), (file2_id, rev1, 'file2')])}
    self.assertEqual(b'foo', extracted['file1-old'])
    self.assertEqual(b'bar', extracted['file2'])
    self.assertEqual(b'baz', extracted['file1-new'])
    self.assertRaises(errors.RevisionNotPresent, list, repository.iter_files_bytes([(file1_id, b'rev3', 'file1-notpresent')]))
    self.assertRaises((errors.RevisionNotPresent, errors.NoSuchId), list, repository.iter_files_bytes([(b'file3-id', b'rev3', 'file1-notpresent')]))