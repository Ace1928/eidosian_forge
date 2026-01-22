from breezy import errors, osutils, repository
from breezy.bzr import btree_index
from breezy.bzr.remote import RemoteRepository
from breezy.bzr.tests.per_repository_chk import TestCaseWithRepositoryCHK
from breezy.bzr.versionedfile import VersionedFiles
from breezy.tests import TestNotApplicable
def make_branch_with_multiple_chk_nodes(self):
    builder = self.make_branch_builder('simple-branch')
    file_adds = []
    file_modifies = []
    for char in 'abc':
        name = char * 10000
        file_adds.append(('add', ('file-' + name, ('file-%s-id' % name).encode(), 'file', ('content %s\n' % name).encode())))
        file_modifies.append(('modify', ('file-' + name, ('new content %s\n' % name).encode())))
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None))] + file_adds, revision_id=b'A-id')
    builder.build_snapshot(None, [], revision_id=b'B-id')
    builder.build_snapshot(None, file_modifies, revision_id=b'C-id')
    return builder.get_branch()