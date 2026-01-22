from neutron_lib.plugins.ml2 import api
from neutron_lib.tests import _base as base
def test__supports_port_binding(self):
    self.assertTrue(_MechanismDriver()._supports_port_binding)