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
def assert_wd(self, expected_paths):
    """Assert paths of files and dirs in wd are same as expected_paths."""
    control_dir_rel = os.path.relpath(self.repo._controldir, self.repo.path)
    found_paths = {os.path.normpath(p) for p in flat_walk_dir(self.repo.path) if not p.split(os.sep)[0] == control_dir_rel}
    norm_expected_paths = {os.path.normpath(p) for p in expected_paths}
    self.assertEqual(found_paths, norm_expected_paths)