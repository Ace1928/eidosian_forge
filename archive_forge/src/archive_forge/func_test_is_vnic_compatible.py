from unittest import mock
from neutron_lib.api.definitions import portbindings
from neutron_lib import constants
from neutron_lib.services.qos import base as qos_base
from neutron_lib.services.qos import constants as qos_consts
from neutron_lib.tests import _base
def test_is_vnic_compatible(self):
    self.assertTrue(_make_driver().is_vnic_compatible(portbindings.VNIC_NORMAL))
    self.assertFalse(_make_driver().is_vnic_compatible(portbindings.VNIC_BAREMETAL))