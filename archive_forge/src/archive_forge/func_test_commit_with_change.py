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
def test_commit_with_change(self):
    a = Blob.from_string(b'The Foo\n')
    ta = Tree()
    ta.add(b'somename', 33188, a.id)
    ca = make_commit(tree=ta.id)
    b = Blob.from_string(b'The Bar\n')
    tb = Tree()
    tb.add(b'somename', 33188, b.id)
    cb = make_commit(tree=tb.id, parents=[ca.id])
    self.repo.object_store.add_objects([(a, None), (b, None), (ta, None), (tb, None), (ca, None), (cb, None)])
    outstream = StringIO()
    porcelain.show(self.repo.path, objects=[cb.id], outstream=outstream)
    self.assertMultiLineEqual(outstream.getvalue(), '--------------------------------------------------\ncommit: 2c6b6c9cb72c130956657e1fdae58e5b103744fa\nAuthor: Test Author <test@nodomain.com>\nCommitter: Test Committer <test@nodomain.com>\nDate:   Fri Jan 01 2010 00:00:00 +0000\n\nTest message.\n\ndiff --git a/somename b/somename\nindex ea5c7bf..fd38bcb 100644\n--- a/somename\n+++ b/somename\n@@ -1 +1 @@\n-The Foo\n+The Bar\n')