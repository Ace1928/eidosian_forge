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
def test_for_each_ref_pattern(self):
    versions = porcelain.for_each_ref(self.repo, pattern='refs/tags/v*')
    self.assertEqual([(object_type, tag) for _, object_type, tag in versions], [(b'tag', b'refs/tags/v0.1'), (b'tag', b'refs/tags/v1.0'), (b'tag', b'refs/tags/v1.1')])
    versions = porcelain.for_each_ref(self.repo, pattern='refs/tags/v1.?')
    self.assertEqual([(object_type, tag) for _, object_type, tag in versions], [(b'tag', b'refs/tags/v1.0'), (b'tag', b'refs/tags/v1.1')])