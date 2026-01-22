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
def test_no_head_no_checkout_outstream_errstream_autofallback(self):
    f1_1 = make_object(Blob, data=b'f1')
    commit_spec = [[1]]
    trees = {1: [(b'f1', f1_1), (b'f2', f1_1)]}
    c1, = build_commit_graph(self.repo.object_store, commit_spec, trees)
    self.repo.refs[b'refs/heads/master'] = c1.id
    target_path = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, target_path)
    errstream = porcelain.NoneStream()
    r = porcelain.clone(self.repo.path, target_path, checkout=True, errstream=errstream)
    r.close()