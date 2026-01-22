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
def test_mac_pattern(self):
    base_mac = 'fa:16:3e:00:00:00'
    msg = validators.validate_regex(base_mac, validators.MAC_PATTERN)
    self.assertIsNone(msg)
    base_mac = 'fa:16:3e:4f:00:00'
    msg = validators.validate_regex(base_mac, validators.MAC_PATTERN)
    self.assertIsNone(msg)
    base_mac = '01:16:3e:4f:00:00'
    msg = validators.validate_regex(base_mac, validators.MAC_PATTERN)
    self.assertIsNotNone(msg)
    base_mac = 'a:16:3e:4f:00:00'
    msg = validators.validate_regex(base_mac, validators.MAC_PATTERN)
    self.assertIsNotNone(msg)
    base_mac = 'ffa:16:3e:4f:00:00'
    msg = validators.validate_regex(base_mac, validators.MAC_PATTERN)
    self.assertIsNotNone(msg)
    base_mac = '01163e4f0000'
    msg = validators.validate_regex(base_mac, validators.MAC_PATTERN)
    self.assertIsNotNone(msg)
    base_mac = '01-16-3e-4f-00-00'
    msg = validators.validate_regex(base_mac, validators.MAC_PATTERN)
    self.assertIsNotNone(msg)
    base_mac = '00:16:3:f:00:00'
    msg = validators.validate_regex(base_mac, validators.MAC_PATTERN)
    self.assertIsNotNone(msg)
    base_mac = '12:3:4:5:67:89ab'
    msg = validators.validate_regex(base_mac, validators.MAC_PATTERN)
    self.assertIsNotNone(msg)