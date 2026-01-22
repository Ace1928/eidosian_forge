import os
import shutil
import tempfile
import unittest
import gevent
from gevent import monkey
from dulwich import client, index, objects, repo, server  # noqa: E402
from dulwich.contrib import swift  # noqa: E402
def test_push_remove_branch(self):

    def determine_wants(*args, **kwargs):
        return {'refs/heads/pullr-108': objects.ZERO_SHA, 'refs/heads/master': local_repo.refs['refs/heads/master'], 'refs/heads/mybranch': local_repo.refs['refs/heads/mybranch']}
    self.test_push_multiple_branch()
    local_repo = repo.Repo(self.temp_d)
    tcp_client = client.TCPGitClient(self.server_address, port=self.port)
    tcp_client.send_pack(self.fakerepo, determine_wants, local_repo.generate_pack_data)
    swift_repo = swift.SwiftRepo('fakerepo', self.conf)
    self.assertNotIn('refs/heads/pullr-108', swift_repo.refs.allkeys())