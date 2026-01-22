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
def test_add_file_clrf_conversion(self):
    c = self.repo.get_config()
    c.set('core', 'autocrlf', 'input')
    c.write_to_path()
    fullpath = os.path.join(self.repo.path, 'foo')
    with open(fullpath, 'wb') as f:
        f.write(b'line1\r\nline2')
    porcelain.add(self.repo.path, paths=[fullpath])
    index = self.repo.open_index()
    self.assertIn(b'foo', index)
    entry = index[b'foo']
    blob = self.repo[entry.sha]
    self.assertEqual(blob.data, b'line1\nline2')