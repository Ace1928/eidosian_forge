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
def test_iso8601(self):
    self.put_envs('1995-11-20T19:12:08-0501')
    self.assertTupleEqual((-18060, -18060), porcelain.get_user_timezones())
    self.put_envs('1995-11-20T19:12:08+0501')
    self.assertTupleEqual((18060, 18060), porcelain.get_user_timezones())
    self.put_envs('1995-11-20T19:12:08-05:01')
    self.assertTupleEqual((-18060, -18060), porcelain.get_user_timezones())
    self.put_envs('1995-11-20 19:12:08-05')
    self.assertTupleEqual((-18000, -18000), porcelain.get_user_timezones())
    self.put_envs('2006-07-03 17:18:44 +0200')
    self.assertTupleEqual((7200, 7200), porcelain.get_user_timezones())