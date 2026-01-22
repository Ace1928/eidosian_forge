import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def remove_from_queue(self, confid):
    """ Removes the candidate confid from the queue. """
    queued_ids = self.c.select(queued=1, gaid=confid)
    ids = [q.id for q in queued_ids]
    self.c.delete(ids)