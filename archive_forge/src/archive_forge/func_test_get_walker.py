import glob
import locale
import os
import shutil
import stat
import sys
import tempfile
import warnings
from dulwich import errors, objects, porcelain
from dulwich.tests import TestCase, skipIf
from ..config import Config
from ..errors import NotGitRepository
from ..object_store import tree_lookup_path
from ..repo import (
from .utils import open_repo, setup_warning_catcher, tear_down_repo
import sys
from dulwich.repo import Repo
def test_get_walker(self):
    r = self.open_repo('a.git')
    self.assertEqual([e.commit.id for e in r.get_walker()], [r.head(), b'2a72d929692c41d8554c07f6301757ba18a65d91'])
    self.assertEqual([e.commit.id for e in r.get_walker([b'2a72d929692c41d8554c07f6301757ba18a65d91'])], [b'2a72d929692c41d8554c07f6301757ba18a65d91'])
    self.assertEqual([e.commit.id for e in r.get_walker(b'2a72d929692c41d8554c07f6301757ba18a65d91')], [b'2a72d929692c41d8554c07f6301757ba18a65d91'])