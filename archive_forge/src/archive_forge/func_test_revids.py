import os
import stat
from dulwich.objects import Blob, Commit, Tree
from ...revision import Revision
from ...tests import TestCase, TestCaseInTempDir, UnavailableFeature
from ...transport import get_transport
from ..cache import (DictBzrGitCache, IndexBzrGitCache, IndexGitCacheFormat,
def test_revids(self):
    self.map.start_write_group()
    updater = self.cache.get_updater(Revision(b'myrevid'))
    c = self._get_test_commit()
    updater.add_object(c, {'testament3-sha1': b'mtestament'}, None)
    updater.finish()
    self.map.commit_write_group()
    self.assertEqual([b'myrevid'], list(self.map.revids()))