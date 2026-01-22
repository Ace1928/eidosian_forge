from neutronclient.common import exceptions as qe
from heat.common import exception
from heat.engine import attributes
from heat.engine import properties
from heat.engine.resources.openstack.neutron import net
from heat.engine.resources.openstack.neutron import neutron as nr
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_is_built(self):
    self.assertTrue(nr.NeutronResource.is_built({'status': 'ACTIVE'}))
    self.assertTrue(nr.NeutronResource.is_built({'status': 'DOWN'}))
    self.assertFalse(nr.NeutronResource.is_built({'status': 'BUILD'}))
    e = self.assertRaises(exception.ResourceInError, nr.NeutronResource.is_built, {'status': 'ERROR'})
    self.assertEqual('Went to status ERROR due to "Unknown"', str(e))
    e = self.assertRaises(exception.ResourceUnknownStatus, nr.NeutronResource.is_built, {'status': 'FROBULATING'})
    self.assertEqual('Resource is not built - Unknown status FROBULATING due to "Unknown"', str(e))