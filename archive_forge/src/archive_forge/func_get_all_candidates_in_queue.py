import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def get_all_candidates_in_queue(self):
    """ Returns all structures that are queued, but have not yet
            been relaxed. """
    all_queued_ids = [t.gaid for t in self.c.select(queued=1)]
    all_relaxed_ids = [t.gaid for t in self.c.select(relaxed=1)]
    in_queue = [qid for qid in all_queued_ids if qid not in all_relaxed_ids]
    return in_queue