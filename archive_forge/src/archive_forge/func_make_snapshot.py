from breezy import errors, revision, tests
from breezy.tests import per_branch
def make_snapshot(self, builder, parents, revid_name):
    self.assertNotIn(revid_name, self.revids)
    if parents is None:
        files = [('add', ('', None, 'directory', ''))]
    else:
        parents = [self.revids[name] for name in parents]
        files = []
    self.revids[revid_name] = builder.build_snapshot(parents, files, message='Revision %s' % revid_name)