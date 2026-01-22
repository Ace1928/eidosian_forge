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
def test_from_subdir(self):
    self.put_files(tracked={'tracked_file', 'tracked_dir/tracked_file', '.gitignore'}, ignored={'ignored_file'}, untracked={'untracked_file', 'tracked_dir/untracked_dir/untracked_file', 'untracked_dir/untracked_dir/untracked_file'}, empty_dirs={'empty_dir'})
    porcelain.clean(repo=self.repo, target_dir=os.path.join(self.repo.path, 'untracked_dir'))
    self.assert_wd({'tracked_file', 'tracked_dir/tracked_file', '.gitignore', 'ignored_file', 'untracked_file', 'tracked_dir/untracked_dir/untracked_file', 'empty_dir', 'untracked_dir', 'tracked_dir', 'tracked_dir/untracked_dir'})