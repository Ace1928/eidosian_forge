from unittest import mock
from neutronclient.v2_0 import client as neutronclient
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import extraroute
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_extraroute_ipv4(self):
    self._test_extraroute(ipv6=False)