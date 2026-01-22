import os
import shutil
import tempfile
import unittest
import gevent
from gevent import monkey
from dulwich import client, index, objects, repo, server  # noqa: E402
from dulwich.contrib import swift  # noqa: E402
def test_clone_bare(self):
    local_repo = repo.Repo.init(self.temp_d, mkdir=True)
    swift.SwiftRepo.init_bare(self.scon, self.conf)
    tcp_client = client.TCPGitClient(self.server_address, port=self.port)
    remote_refs = tcp_client.fetch(self.fakerepo, local_repo)
    self.assertEqual(remote_refs, None)