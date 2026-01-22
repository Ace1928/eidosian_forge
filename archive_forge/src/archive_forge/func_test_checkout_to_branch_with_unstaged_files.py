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
def test_checkout_to_branch_with_unstaged_files(self):
    with open(self._foo_path, 'a') as f:
        f.write('new message')
    status = list(porcelain.status(self.repo))
    self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [b'foo'], []], status)
    porcelain.checkout_branch(self.repo, b'uni')
    status = list(porcelain.status(self.repo))
    self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [b'foo'], []], status)