from breezy.tests.per_controldir import TestCaseWithControlDir
from ...controldir import NoColocatedBranchSupport
from ...errors import LossyPushToSameVCS, NoSuchRevision, TagsNotSupported
from ...revision import NULL_REVISION
from .. import TestNotApplicable
def test_push_new_branch_fetch_tags(self):
    builder = self.make_branch_builder('from')
    builder.start_series()
    rev_1 = builder.build_snapshot(None, [('add', ('', None, 'directory', '')), ('add', ('filename', None, 'file', b'content'))])
    rev_2 = builder.build_snapshot([rev_1], [('modify', ('filename', b'new-content\n'))])
    rev_3 = builder.build_snapshot([rev_1], [('modify', ('filename', b'new-new-content\n'))])
    builder.finish_series()
    branch = builder.get_branch()
    try:
        branch.tags.set_tag('atag', rev_2)
    except TagsNotSupported:
        raise TestNotApplicable('source format does not support tags')
    dir = self.make_repository('target').controldir
    branch.get_config().set_user_option('branch.fetch_tags', True)
    result = dir.push_branch(branch)
    self.assertEqual({rev_1, rev_2, rev_3}, set(result.source_branch.repository.all_revision_ids()))
    self.assertEqual({'atag': rev_2}, result.source_branch.tags.get_tag_dict())