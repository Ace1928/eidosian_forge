from .. import errors, osutils
from .. import revision as _mod_revision
from ..branch import Branch
from ..bzr import bzrdir, knitrepo, versionedfile
from ..upgrade import Convert
from ..workingtree import WorkingTree
from . import TestCaseWithTransport
from .test_revision import make_branches
def test_fetch_with_fallback_and_merge(self):
    builder = self.make_branch_builder('source', format='pack-0.92')
    builder.start_series()
    to_add = [('add', ('', b'TREE_ROOT', 'directory', None))]
    for i in range(10):
        fname = 'file%03d' % (i,)
        fileid = ('%s-%s' % (fname, osutils.rand_chars(64))).encode('ascii')
        to_add.append(('add', (fname, fileid, 'file', b'content\n')))
    builder.build_snapshot(None, to_add, revision_id=b'A')
    builder.build_snapshot([b'A'], [], revision_id=b'B')
    builder.build_snapshot([b'A'], [], revision_id=b'C')
    builder.build_snapshot([b'C'], [], revision_id=b'D')
    builder.build_snapshot([b'D'], [], revision_id=b'E')
    builder.build_snapshot([b'E', b'B'], [], revision_id=b'F')
    builder.finish_series()
    source_branch = builder.get_branch()
    source_branch.controldir.sprout('base', revision_id=b'B')
    target_branch = self.make_branch('target', format='1.6')
    target_branch.set_stacked_on_url('../base')
    source = source_branch.repository
    source.lock_read()
    self.addCleanup(source.unlock)
    source.inventories = versionedfile.OrderingVersionedFilesDecorator(source.inventories, key_priority={(b'E',): 1, (b'D',): 2, (b'C',): 4, (b'F',): 3})
    records = [(record.key, record.storage_kind) for record in source.inventories.get_record_stream([(b'D',), (b'C',), (b'E',), (b'F',)], 'unordered', False)]
    self.assertEqual([((b'E',), 'knit-delta-gz'), ((b'D',), 'knit-delta-gz'), ((b'F',), 'knit-delta-gz'), ((b'C',), 'knit-delta-gz')], records)
    target_branch.lock_write()
    self.addCleanup(target_branch.unlock)
    target = target_branch.repository
    target.fetch(source, revision_id=b'F')
    stream = target.inventories.get_record_stream([(b'C',), (b'D',), (b'E',), (b'F',)], 'unordered', False)
    kinds = {record.key: record.storage_kind for record in stream}
    self.assertEqual({(b'C',): 'knit-ft-gz', (b'D',): 'knit-delta-gz', (b'E',): 'knit-delta-gz', (b'F',): 'knit-delta-gz'}, kinds)