import string
from unittest import mock
import netaddr
from neutron_lib._i18n import _
from neutron_lib.api import converters
from neutron_lib.api.definitions import extra_dhcp_opt
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib import fixture
from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
def test_validate_ip_pools(self):
    pools = [[{'end': '10.0.0.254'}], [{'start': '10.0.0.254'}], [{'start': '1000.0.0.254', 'end': '1.1.1.1'}], [{'start': '10.0.0.2', 'end': '10.0.0.254', 'forza': 'juve'}], [{'start': '10.0.0.2', 'end': '10.0.0.254'}, {'end': '10.0.0.254'}], [None], None]
    for pool in pools:
        msg = validators.validate_ip_pools(pool)
        self.assertIsNotNone(msg)
    pools = [[{'end': '10.0.0.254', 'start': '10.0.0.2'}, {'start': '11.0.0.2', 'end': '11.1.1.1'}], [{'start': '11.0.0.2', 'end': '11.0.0.100'}]]
    for pool in pools:
        msg = validators.validate_ip_pools(pool)
        self.assertIsNone(msg)
    invalid_ip = '10.0.0.2\r'
    pools = [[{'end': '10.0.0.254', 'start': invalid_ip}]]
    for pool in pools:
        msg = validators.validate_ip_pools(pool)
        self.assertEqual("'%s' is not a valid IP address" % invalid_ip, msg)