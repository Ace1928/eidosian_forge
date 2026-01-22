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
def test_reset_modify_file_to_commit(self):
    file = 'foo'
    full_path = os.path.join(self.repo.path, file)
    with open(full_path, 'w') as f:
        f.write('hello')
    porcelain.add(self.repo, paths=[full_path])
    sha = porcelain.commit(self.repo, message=b'unitest', committer=b'Jane <jane@example.com>', author=b'John <john@example.com>')
    with open(full_path, 'a') as f:
        f.write('something new')
    porcelain.reset_file(self.repo, file, target=sha)
    with open(full_path) as f:
        self.assertEqual('hello', f.read())