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
def test_fetch_empty_pack(self):
    c = self._client()
    with repo.Repo(os.path.join(self.gitroot, 'dest')) as dest:
        result = c.fetch(self._build_path('/server_new.export'), dest)
        for r in result.refs.items():
            dest.refs.set_if_equals(r[0], None, r[1])
        self.assertDestEqualsSrc()

        def dw(refs, **kwargs):
            return list(refs.values())
        result = c.fetch(self._build_path('/server_new.export'), dest, determine_wants=dw)
        for r in result.refs.items():
            dest.refs.set_if_equals(r[0], None, r[1])
        self.assertDestEqualsSrc()