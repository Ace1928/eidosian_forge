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
def test_get_tree_changes_delete(self):
    """Unit test for get_tree_changes delete."""
    filename = 'foo'
    fullpath = os.path.join(self.repo.path, filename)
    with open(fullpath, 'w') as f:
        f.write('stuff')
    porcelain.add(repo=self.repo.path, paths=fullpath)
    porcelain.commit(repo=self.repo.path, message=b'test status', author=b'author <email>', committer=b'committer <email>')
    cwd = os.getcwd()
    try:
        os.chdir(self.repo.path)
        porcelain.remove(repo=self.repo.path, paths=[filename])
    finally:
        os.chdir(cwd)
    changes = porcelain.get_tree_changes(self.repo.path)
    self.assertEqual(changes['delete'][0], filename.encode('ascii'))
    self.assertEqual(len(changes['add']), 0)
    self.assertEqual(len(changes['modify']), 0)
    self.assertEqual(len(changes['delete']), 1)