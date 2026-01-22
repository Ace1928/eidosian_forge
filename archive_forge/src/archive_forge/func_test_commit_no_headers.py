import json
from openstack.object_store.v1 import container
from openstack.tests.unit import base
def test_commit_no_headers(self):
    sot = container.Container.new(name=self.container)
    self._test_no_headers(sot, sot.commit, 'POST')
    self.assert_calls()