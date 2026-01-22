import hashlib
from unittest import mock
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import uuidutils
from neutron_lib.api.definitions import portbindings_extended as pb_ext
from neutron_lib import constants
from neutron_lib import exceptions
from neutron_lib.plugins import utils
from neutron_lib.tests import _base as base
def test_get_interface_name(self):
    prefix = 'pre-'
    prefix_long = 'long_prefix'
    prefix_exceeds_max_dev_len = 'much_too_long_prefix'
    self.assertEqual('A_REALLY_' + self._hash_prefix(LONG_NAME1), utils.get_interface_name(LONG_NAME1))
    self.assertEqual('SHORT', utils.get_interface_name(SHORT_NAME))
    self.assertEqual('pre-A_REA' + self._hash_prefix(LONG_NAME1), utils.get_interface_name(LONG_NAME1, prefix=prefix))
    self.assertEqual('pre-SHORT', utils.get_interface_name(SHORT_NAME, prefix=prefix))
    self.assertRaises(ValueError, utils.get_interface_name, SHORT_NAME, prefix_long)
    self.assertRaises(ValueError, utils.get_interface_name, SHORT_NAME, prefix=prefix_exceeds_max_dev_len)