import copy
from unittest import mock
from heat.common import exception
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_chain
from heat.engine import rsrc_defn
from heat.objects import service as service_objects
from heat.tests import common
from heat.tests import utils
def test_index_refs(self):
    """Tests getting ids of individual resources."""
    chain = self._create_dummy_stack()
    self.assertEqual('ID-0', chain.FnGetAtt('resource.0'))
    self.assertEqual('ID-1', chain.FnGetAtt('resource.1'))
    ex = self.assertRaises(exception.NotFound, chain.FnGetAtt, 'resource.2')
    self.assertIn("Member '2' not found in group resource 'test'", str(ex))