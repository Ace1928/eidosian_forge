import collections
from unittest import mock
import uuid
from novaclient import client as nc
from novaclient import exceptions as nova_exceptions
from oslo_config import cfg
from oslo_serialization import jsonutils as json
import requests
from heat.common import exception
from heat.engine.clients.os import nova
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_absolute_limits_failure(self):
    limits = mock.Mock()
    limits.absolute = self._absolute_limits()
    self.nova_client.limits.get.side_effect = [requests.ConnectionError, requests.ConnectionError, requests.ConnectionError]
    self.assertRaises(requests.ConnectionError, self.nova_plugin.absolute_limits)