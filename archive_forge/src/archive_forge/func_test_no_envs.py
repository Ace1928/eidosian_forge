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
def test_no_envs(self):
    local_timezone = time.localtime().tm_gmtoff
    self.put_envs('0 +0500')
    self.assertTupleEqual((18000, 18000), porcelain.get_user_timezones())
    self.overrideEnv('GIT_COMMITTER_DATE', None)
    self.assertTupleEqual((18000, local_timezone), porcelain.get_user_timezones())
    self.put_envs('0 +0500')
    self.overrideEnv('GIT_AUTHOR_DATE', None)
    self.assertTupleEqual((local_timezone, 18000), porcelain.get_user_timezones())
    self.put_envs('0 +0500')
    self.overrideEnv('GIT_AUTHOR_DATE', None)
    self.overrideEnv('GIT_COMMITTER_DATE', None)
    self.assertTupleEqual((local_timezone, local_timezone), porcelain.get_user_timezones())