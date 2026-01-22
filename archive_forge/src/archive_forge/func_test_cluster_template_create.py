import copy
from unittest import mock
from neutronclient.neutron import v2_0 as neutronV20
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.magnum import cluster_template
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_cluster_template_create(self):
    bm = self._create_resource('bm', self.rsrc_defn, self.stack)
    self.assertEqual(self.resource_id, bm.resource_id)
    self.assertEqual((bm.CREATE, bm.COMPLETE), bm.state)
    self.client.cluster_templates.create.assert_called_once_with(**self.expected)