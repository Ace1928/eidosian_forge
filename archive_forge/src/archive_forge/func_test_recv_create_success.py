from unittest import mock
from openstack import exceptions
from heat.common import template_format
from heat.engine.clients.os import senlin
from heat.engine.resources.openstack.senlin import receiver as sr
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_recv_create_success(self):
    self._create_recv(self.t)
    expect_kwargs = {'name': 'SenlinReceiver', 'cluster_id': 'fake_cluster', 'action': 'CLUSTER_SCALE_OUT', 'type': 'webhook', 'params': {'foo': 'bar'}}
    self.senlin_mock.create_receiver.assert_called_once_with(**expect_kwargs)