import os
import subprocess
import sys
from io import BytesIO
from dulwich.repo import Repo
from ...tests import TestCaseWithTransport
from ...tests.features import PathFeature
from ..git_remote_helper import RemoteHelper, fetch, open_local_dir
from ..object_store import get_object_store
from . import FastimportFeature
def test_list_basic(self):
    f = BytesIO()
    self.helper.cmd_list(f, [])
    self.assertEqual(b'\n', f.getvalue())