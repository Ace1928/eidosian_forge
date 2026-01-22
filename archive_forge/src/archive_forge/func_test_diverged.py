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
def test_diverged(self):
    outstream = BytesIO()
    errstream = BytesIO()
    c3a = porcelain.commit(repo=self.target_path, message=b'test3a', author=b'test2 <email>', committer=b'test2 <email>')
    porcelain.commit(repo=self.repo.path, message=b'test3b', author=b'test2 <email>', committer=b'test2 <email>')
    self.assertRaises(porcelain.DivergedBranches, porcelain.pull, self.target_path, self.repo.path, b'refs/heads/master', outstream=outstream, errstream=errstream)
    with Repo(self.target_path) as r:
        self.assertEqual(r[b'refs/heads/master'].id, c3a)
    self.assertRaises(NotImplementedError, porcelain.pull, self.target_path, self.repo.path, b'refs/heads/master', outstream=outstream, errstream=errstream, fast_forward=False)
    with Repo(self.target_path) as r:
        self.assertEqual(r[b'refs/heads/master'].id, c3a)