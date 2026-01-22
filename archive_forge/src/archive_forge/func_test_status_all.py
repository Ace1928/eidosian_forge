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
def test_status_all(self):
    del_path = os.path.join(self.repo.path, 'foo')
    mod_path = os.path.join(self.repo.path, 'bar')
    add_path = os.path.join(self.repo.path, 'baz')
    us_path = os.path.join(self.repo.path, 'blye')
    ut_path = os.path.join(self.repo.path, 'blyat')
    with open(del_path, 'w') as f:
        f.write('origstuff')
    with open(mod_path, 'w') as f:
        f.write('origstuff')
    with open(us_path, 'w') as f:
        f.write('origstuff')
    porcelain.add(repo=self.repo.path, paths=[del_path, mod_path, us_path])
    porcelain.commit(repo=self.repo.path, message=b'test status', author=b'author <email>', committer=b'committer <email>')
    porcelain.remove(self.repo.path, [del_path])
    with open(add_path, 'w') as f:
        f.write('origstuff')
    with open(mod_path, 'w') as f:
        f.write('more_origstuff')
    with open(us_path, 'w') as f:
        f.write('more_origstuff')
    porcelain.add(repo=self.repo.path, paths=[add_path, mod_path])
    with open(us_path, 'w') as f:
        f.write('\norigstuff')
    with open(ut_path, 'w') as f:
        f.write('origstuff')
    results = porcelain.status(self.repo.path)
    self.assertDictEqual({'add': [b'baz'], 'delete': [b'foo'], 'modify': [b'bar']}, results.staged)
    self.assertListEqual(results.unstaged, [b'blye'])
    results_no_untracked = porcelain.status(self.repo.path, untracked_files='no')
    self.assertListEqual(results_no_untracked.untracked, [])