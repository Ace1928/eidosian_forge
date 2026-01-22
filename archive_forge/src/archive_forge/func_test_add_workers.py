import abc
from neutron_lib.services import base
from neutron_lib.tests import _base as test_base
def test_add_workers(self):
    workers = [object(), object(), object()]
    self.worker.add_workers(workers)
    self.assertSequenceEqual(workers, self.worker.get_workers())