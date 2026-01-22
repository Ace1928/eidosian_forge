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
def test_with_remote_name(self):
    remote_name = 'origin'
    outstream = BytesIO()
    errstream = BytesIO()
    handle, fullpath = tempfile.mkstemp(dir=self.repo.path)
    os.close(handle)
    porcelain.add(repo=self.repo.path, paths=fullpath)
    porcelain.commit(repo=self.repo.path, message=b'test', author=b'test <email>', committer=b'test <email>')
    target_path = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, target_path)
    target_repo = porcelain.clone(self.repo.path, target=target_path, errstream=errstream)
    target_refs = target_repo.get_refs()
    handle, fullpath = tempfile.mkstemp(dir=self.repo.path)
    os.close(handle)
    porcelain.add(repo=self.repo.path, paths=fullpath)
    porcelain.commit(repo=self.repo.path, message=b'test2', author=b'test2 <email>', committer=b'test2 <email>')
    self.assertNotIn(self.repo[b'HEAD'].id, target_repo)
    target_config = target_repo.get_config()
    target_config.set((b'remote', remote_name.encode()), b'url', self.repo.path.encode())
    target_repo.close()
    porcelain.fetch(target_path, remote_name, outstream=outstream, errstream=errstream)
    self.assert_correct_remote_refs(target_repo.get_refs(), self.repo.get_refs())
    with Repo(target_path) as r:
        self.assertIn(self.repo[b'HEAD'].id, r)
        self.assertNotEqual(self.repo.get_refs(), target_refs)