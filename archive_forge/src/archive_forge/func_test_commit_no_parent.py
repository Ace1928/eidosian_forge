import contextlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from io import BytesIO, StringIO
from unittest import skipIf
from dulwich import porcelain
from dulwich.tests import TestCase
from ..diff_tree import tree_changes
from ..errors import CommitError
from ..objects import ZERO_SHA, Blob, Tag, Tree
from ..porcelain import CheckoutError
from ..repo import NoIndexPresent, Repo
from ..server import DictBackend
from ..web import make_server, make_wsgi_chain
from .utils import build_commit_graph, make_commit, make_object
def test_commit_no_parent(self):
    a = Blob.from_string(b'The Foo\n')
    ta = Tree()
    ta.add(b'somename', 33188, a.id)
    ca = make_commit(tree=ta.id)
    self.repo.object_store.add_objects([(a, None), (ta, None), (ca, None)])
    outstream = StringIO()
    porcelain.show(self.repo.path, objects=[ca.id], outstream=outstream)
    self.assertMultiLineEqual(outstream.getvalue(), '--------------------------------------------------\ncommit: 344da06c1bb85901270b3e8875c988a027ec087d\nAuthor: Test Author <test@nodomain.com>\nCommitter: Test Committer <test@nodomain.com>\nDate:   Fri Jan 01 2010 00:00:00 +0000\n\nTest message.\n\ndiff --git a/somename b/somename\nnew file mode 100644\nindex 0000000..ea5c7bf\n--- /dev/null\n+++ b/somename\n@@ -0,0 +1 @@\n+The Foo\n')