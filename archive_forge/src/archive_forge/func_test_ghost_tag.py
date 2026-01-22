from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_ghost_tag(self):
    b = self.make_branch('b')
    if not b._format.supports_tags_referencing_ghosts():
        self.assertRaises(errors.GhostTagsNotSupported, b.tags.set_tag, 'ghost', b'idontexist')
    else:
        b.tags.set_tag('ghost', b'idontexist')
        self.assertEqual(b'idontexist', b.tags.lookup_tag('ghost'))