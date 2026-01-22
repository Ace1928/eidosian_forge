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
def test_out_of_order_merge(self):
    """Test that revision history is ordered by date, not parent order."""
    r = self.open_repo('ooo_merge.git')
    shas = [e.commit.id for e in r.get_walker()]
    self.assertEqual(shas, [b'7601d7f6231db6a57f7bbb79ee52e4d462fd44d1', b'f507291b64138b875c28e03469025b1ea20bc614', b'fb5b0425c7ce46959bec94d54b9a157645e114f5', b'f9e39b120c68182a4ba35349f832d0e4e61f485c'])