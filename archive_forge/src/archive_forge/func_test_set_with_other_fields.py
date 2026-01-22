from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_set_with_other_fields(self):
    self.node.is_maintenance = True
    self.node.name = 'lazy-3000'
    self.node.commit(self.session)
    self.session.put.assert_called_once_with('nodes/%s/maintenance' % self.node.id, json={'reason': None}, headers=mock.ANY, microversion=mock.ANY)
    self.session.patch.assert_called_once_with('nodes/%s' % self.node.id, json=[{'path': '/name', 'op': 'replace', 'value': 'lazy-3000'}], headers=mock.ANY, microversion=mock.ANY)