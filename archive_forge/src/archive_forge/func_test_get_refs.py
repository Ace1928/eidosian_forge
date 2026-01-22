import copy
import http.server
import os
import select
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
from contextlib import suppress
from io import BytesIO
from urllib.parse import unquote
from dulwich import client, file, index, objects, protocol, repo
from dulwich.tests import SkipTest, expectedFailure
from .utils import (
def test_get_refs(self):
    c = self._client()
    refs = c.get_refs(self._build_path('/server_new.export'))
    repo_dir = os.path.join(self.gitroot, 'server_new.export')
    with repo.Repo(repo_dir) as dest:
        self.assertDictEqual(dest.refs.as_dict(), refs)