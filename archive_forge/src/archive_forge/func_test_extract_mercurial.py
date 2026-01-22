from io import BytesIO, StringIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..patch import (
def test_extract_mercurial(self):
    raise SkipTest("git_am_patch_split doesn't handle Mercurial patches properly yet")
    expected_diff = "diff --git a/dulwich/tests/test_patch.py b/dulwich/tests/test_patch.py\n--- a/dulwich/tests/test_patch.py\n+++ b/dulwich/tests/test_patch.py\n@@ -158,7 +158,7 @@\n \n '''\n         c, diff, version = git_am_patch_split(BytesIO(text))\n-        self.assertIs(None, version)\n+        self.assertEqual(None, version)\n \n \n class DiffTests(TestCase):\n"
    text = 'From dulwich-users-bounces+jelmer=samba.org@lists.launchpad.net Mon Nov 29 00:58:18 2010\nDate: Sun, 28 Nov 2010 17:57:27 -0600\nFrom: Augie Fackler <durin42@gmail.com>\nTo: dulwich-users <dulwich-users@lists.launchpad.net>\nSubject: [Dulwich-users] [PATCH] test_patch: fix tests on Python 2.6\nContent-Transfer-Encoding: 8bit\n\nChange-Id: I5e51313d4ae3a65c3f00c665002a7489121bb0d6\n\n%s\n\n_______________________________________________\nMailing list: https://launchpad.net/~dulwich-users\nPost to     : dulwich-users@lists.launchpad.net\nUnsubscribe : https://launchpad.net/~dulwich-users\nMore help   : https://help.launchpad.net/ListHelp\n\n' % expected_diff
    c, diff, version = git_am_patch_split(BytesIO(text))
    self.assertEqual(expected_diff, diff)
    self.assertEqual(None, version)