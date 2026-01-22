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
def test_remove_file(self):
    fullpath = os.path.join(self.repo.path, 'foo')
    with open(fullpath, 'w') as f:
        f.write('BAR')
    porcelain.add(self.repo.path, paths=[fullpath])
    porcelain.commit(repo=self.repo, message=b'test', author=b'test <email>', committer=b'test <email>')
    self.assertTrue(os.path.exists(os.path.join(self.repo.path, 'foo')))
    cwd = os.getcwd()
    try:
        os.chdir(self.repo.path)
        porcelain.remove(self.repo.path, paths=['foo'])
    finally:
        os.chdir(cwd)
    self.assertFalse(os.path.exists(os.path.join(self.repo.path, 'foo')))