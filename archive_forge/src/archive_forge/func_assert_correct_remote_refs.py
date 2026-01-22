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
def assert_correct_remote_refs(self, local_refs, remote_refs, remote_name=b'origin'):
    """Assert that known remote refs corresponds to actual remote refs."""
    local_ref_prefix = b'refs/heads'
    remote_ref_prefix = b'refs/remotes/' + remote_name
    locally_known_remote_refs = {k[len(remote_ref_prefix) + 1:]: v for k, v in local_refs.items() if k.startswith(remote_ref_prefix)}
    normalized_remote_refs = {k[len(local_ref_prefix) + 1:]: v for k, v in remote_refs.items() if k.startswith(local_ref_prefix)}
    if b'HEAD' in locally_known_remote_refs and b'HEAD' in remote_refs:
        normalized_remote_refs[b'HEAD'] = remote_refs[b'HEAD']
    self.assertEqual(locally_known_remote_refs, normalized_remote_refs)