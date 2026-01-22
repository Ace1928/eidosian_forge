from ..fetch_ghosts import GhostFetcher
from . import TestCaseWithTransport
def prepare_with_ghosts(self):
    tree = self.make_branch_and_tree('.')
    tree.commit('rev1', rev_id=b'rev1-id')
    tree.set_parent_ids([b'rev1-id', b'ghost-id'])
    tree.commit('rev2')
    return tree