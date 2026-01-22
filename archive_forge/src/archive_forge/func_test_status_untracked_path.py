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
def test_status_untracked_path(self):
    untracked_dir = os.path.join(self.repo_path, 'untracked_dir')
    os.mkdir(untracked_dir)
    untracked_file = os.path.join(untracked_dir, 'untracked_file')
    with open(untracked_file, 'w') as fh:
        fh.write('untracked')
    _, _, untracked = porcelain.status(self.repo.path, untracked_files='all')
    self.assertEqual(untracked, ['untracked_dir/untracked_file'])