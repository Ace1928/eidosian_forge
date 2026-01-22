from stat import S_ISDIR
from ... import controldir, errors, gpg, osutils, repository
from ... import revision as _mod_revision
from ... import tests, transport, ui
from ...tests import TestCaseWithTransport, TestNotApplicable, test_server
from ...transport import memory
from .. import inventory
from ..btree_index import BTreeGraphIndex
from ..groupcompress_repo import RepositoryFormat2a
from ..index import GraphIndex
from ..smart import client
def test_commit_resumed_write_group(self):
    self.vfs_transport_factory = memory.MemoryServer
    repo = self.make_repository('repo', format=self.get_format())
    token = self._lock_write(repo).repository_token
    repo.start_write_group()
    text_key = (b'file-id', b'revid')
    repo.texts.add_lines(text_key, (), [b'lines'])
    wg_tokens = repo.suspend_write_group()
    same_repo = repo.controldir.open_repository()
    same_repo.lock_write()
    self.addCleanup(same_repo.unlock)
    same_repo.resume_write_group(wg_tokens)
    same_repo.commit_write_group()
    expected_pack_name = wg_tokens[0] + '.pack'
    expected_names = [wg_tokens[0] + ext for ext in ('.rix', '.iix', '.tix', '.six')]
    if repo.chk_bytes is not None:
        expected_names.append(wg_tokens[0] + '.cix')
    self.assertEqual([], same_repo._pack_collection._upload_transport.list_dir(''))
    index_names = repo._pack_collection._index_transport.list_dir('')
    self.assertEqual(sorted(expected_names), sorted(index_names))
    pack_names = repo._pack_collection._pack_transport.list_dir('')
    self.assertEqual([expected_pack_name], pack_names)