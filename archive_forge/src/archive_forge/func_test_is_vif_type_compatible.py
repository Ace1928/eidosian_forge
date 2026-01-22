from unittest import mock
from neutron_lib.api.definitions import portbindings
from neutron_lib import constants
from neutron_lib.services.qos import base as qos_base
from neutron_lib.services.qos import constants as qos_consts
from neutron_lib.tests import _base
def test_is_vif_type_compatible(self):
    self.assertTrue(_make_driver().is_vif_type_compatible(portbindings.VIF_TYPE_OVS))
    self.assertFalse(_make_driver().is_vif_type_compatible(portbindings.VIF_TYPE_BRIDGE))