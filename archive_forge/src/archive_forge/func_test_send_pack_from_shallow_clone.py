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
def test_send_pack_from_shallow_clone(self):
    c = self._client()
    server_new_path = os.path.join(self.gitroot, 'server_new.export')
    run_git_or_fail(['config', 'http.uploadpack', 'true'], cwd=server_new_path)
    run_git_or_fail(['config', 'http.receivepack', 'true'], cwd=server_new_path)
    remote_path = self._build_path('/server_new.export')
    with repo.Repo(self.dest) as local:
        result = c.fetch(remote_path, local, depth=1)
        for r in result.refs.items():
            local.refs.set_if_equals(r[0], None, r[1])
        tree_id = local[local.head()].tree
        for filename, contents in [('bar', 'bar contents'), ('zop', 'zop contents')]:
            tree_id = self._add_file(local, tree_id, filename, contents)
            commit_id = local.do_commit(message=b'add ' + filename.encode('utf-8'), committer=b'Joe Example <joe@example.com>', tree=tree_id)
        sendrefs = dict(local.get_refs())
        del sendrefs[b'HEAD']
        c.send_pack(remote_path, lambda _: sendrefs, local.generate_pack_data)
    with repo.Repo(server_new_path) as remote:
        self.assertEqual(remote.head(), commit_id)