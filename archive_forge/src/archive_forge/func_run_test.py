from dulwich.tests import TestCase
from ..graph import WorkList, _find_lcas, can_fast_forward
from ..repo import MemoryRepo
from .utils import make_commit
@staticmethod
def run_test(dag, inputs):

    def lookup_parents(commit_id):
        return dag[commit_id]

    def lookup_stamp(commit_id):
        return 100
    c1 = inputs[0]
    c2s = inputs[1:]
    return set(_find_lcas(lookup_parents, c1, c2s, lookup_stamp))