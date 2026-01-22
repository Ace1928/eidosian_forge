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
def test_send_without_report_status(self):
    c = self._client()
    c._send_capabilities.remove(b'report-status')
    srcpath = os.path.join(self.gitroot, 'server_new.export')
    with repo.Repo(srcpath) as src:
        sendrefs = dict(src.get_refs())
        del sendrefs[b'HEAD']
        c.send_pack(self._build_path('/dest'), lambda _: sendrefs, src.generate_pack_data)
        self.assertDestEqualsSrc()