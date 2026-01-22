from openstack.network.v2 import bgpvpn as _bgpvpn
from openstack.network.v2 import (
from openstack.network.v2 import bgpvpn_port_association as _bgpvpn_port_assoc
from openstack.network.v2 import (
from openstack.network.v2 import network as _network
from openstack.network.v2 import port as _port
from openstack.network.v2 import router as _router
from openstack.network.v2 import subnet as _subnet
from openstack.tests.functional import base
def test_update_bgpvpn(self):
    sot = self.operator_cloud.network.update_bgpvpn(self.BGPVPN.id, import_targets='64512:1333')
    self.assertEqual(['64512:1333'], sot.import_targets)