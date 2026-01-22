import abc
from neutron_lib.services import base
from neutron_lib.tests import _base as test_base
def test_add_worker(self):
    workers = [object(), object()]
    for w in workers:
        self.worker.add_worker(w)
    self.assertSequenceEqual(workers, self.worker.get_workers())