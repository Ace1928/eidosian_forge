import os
import shutil
import tempfile
import unittest
import gevent
from gevent import monkey
from dulwich import client, index, objects, repo, server  # noqa: E402
from dulwich.contrib import swift  # noqa: E402
def test_push_annotated_tag(self):

    def determine_wants(*args, **kwargs):
        return {'refs/heads/master': local_repo.refs['HEAD'], 'refs/tags/v1.0': local_repo.refs['refs/tags/v1.0']}
    local_repo = repo.Repo.init(self.temp_d, mkdir=True)
    sha = local_repo.do_commit('Test commit', 'fbo@localhost')
    otype, data = local_repo.object_store.get_raw(sha)
    commit = objects.ShaFile.from_raw_string(otype, data)
    tag = objects.Tag()
    tag.tagger = 'fbo@localhost'
    tag.message = 'Annotated tag'
    tag.tag_timezone = objects.parse_timezone('-0200')[0]
    tag.tag_time = commit.author_time
    tag.object = (objects.Commit, commit.id)
    tag.name = 'v0.1'
    local_repo.object_store.add_object(tag)
    local_repo.refs['refs/tags/v1.0'] = tag.id
    swift.SwiftRepo.init_bare(self.scon, self.conf)
    tcp_client = client.TCPGitClient(self.server_address, port=self.port)
    tcp_client.send_pack(self.fakerepo, determine_wants, local_repo.generate_pack_data)
    swift_repo = swift.SwiftRepo(self.fakerepo, self.conf)
    tag_sha = swift_repo.refs.read_loose_ref('refs/tags/v1.0')
    otype, data = swift_repo.object_store.get_raw(tag_sha)
    rtag = objects.ShaFile.from_raw_string(otype, data)
    self.assertEqual(rtag.object[1], commit.id)
    self.assertEqual(rtag.id, tag.id)