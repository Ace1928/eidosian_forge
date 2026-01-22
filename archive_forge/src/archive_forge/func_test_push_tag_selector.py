from breezy.tests.per_controldir import TestCaseWithControlDir
from ...controldir import NoColocatedBranchSupport
from ...errors import LossyPushToSameVCS, NoSuchRevision, TagsNotSupported
from ...revision import NULL_REVISION
from .. import TestNotApplicable
def test_push_tag_selector(self):
    tree, rev1 = self.create_simple_tree()
    try:
        tree.branch.tags.set_tag('tag1', rev1)
    except TagsNotSupported:
        raise TestNotApplicable('tags not supported')
    tree.branch.tags.set_tag('tag2', rev1)
    dir = self.make_repository('dir').controldir
    dir.push_branch(tree.branch, tag_selector=lambda x: x == 'tag1')
    self.assertEqual({'tag1': rev1}, dir.open_branch().tags.get_tag_dict())