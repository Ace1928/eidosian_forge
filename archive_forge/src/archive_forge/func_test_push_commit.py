import os
import shutil
import tempfile
import unittest
import gevent
from gevent import monkey
from dulwich import client, index, objects, repo, server  # noqa: E402
from dulwich.contrib import swift  # noqa: E402
def test_push_commit(self):

    def determine_wants(*args, **kwargs):
        return {'refs/heads/master': local_repo.refs['HEAD']}
    local_repo = repo.Repo.init(self.temp_d, mkdir=True)
    local_repo.do_commit('Test commit', 'fbo@localhost')
    sha = local_repo.refs.read_loose_ref('refs/heads/master')
    swift.SwiftRepo.init_bare(self.scon, self.conf)
    tcp_client = client.TCPGitClient(self.server_address, port=self.port)
    tcp_client.send_pack(self.fakerepo, determine_wants, local_repo.generate_pack_data)
    swift_repo = swift.SwiftRepo('fakerepo', self.conf)
    remote_sha = swift_repo.refs.read_loose_ref('refs/heads/master')
    self.assertEqual(sha, remote_sha)