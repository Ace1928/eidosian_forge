from breezy.tests.per_repository import TestCaseWithRepository
def test_pack_accepts_opaque_hint(self):
    tree = self.make_branch_and_tree('tree')
    rev1 = tree.commit('1')
    rev2 = tree.commit('2')
    rev3 = tree.commit('3')
    rev4 = tree.commit('4')
    tree.branch.repository.pack(hint=[rev3.decode('utf-8'), rev4.decode('utf-8')])