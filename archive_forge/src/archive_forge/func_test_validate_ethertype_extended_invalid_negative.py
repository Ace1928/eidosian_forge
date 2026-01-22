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
@mock.patch('oslo_config.cfg.CONF')
def test_validate_ethertype_extended_invalid_negative(self, CONF):
    CONF.sg_filter_ethertypes = True
    self.assertEqual('Ethertype -16392 is not a two octet hexadecimal number or ethertype name.', validators.validate_ethertype('-0x4008'))