import os
import stat
from dulwich.objects import Blob, Commit, Tree
from ...revision import Revision
from ...tests import TestCase, TestCaseInTempDir, UnavailableFeature
from ...transport import get_transport
from ..cache import (DictBzrGitCache, IndexBzrGitCache, IndexGitCacheFormat,
def test_tree(self):
    self.map.start_write_group()
    updater = self.cache.get_updater(Revision(b'somerevid'))
    updater.add_object(self._get_test_commit(), {'testament3-sha1': b'mytestamentsha'}, None)
    t = Tree()
    t.add(b'somename', stat.S_IFREG, Blob().id)
    updater.add_object(t, (b'fileid', b'myrevid'), b'')
    updater.finish()
    self.map.commit_write_group()
    self.assertEqual([('tree', (b'fileid', b'myrevid'))], list(self.map.lookup_git_sha(t.id)))
    try:
        self.assertEqual(t.id, self.map.lookup_tree_id(b'fileid', b'myrevid'))
    except NotImplementedError:
        pass