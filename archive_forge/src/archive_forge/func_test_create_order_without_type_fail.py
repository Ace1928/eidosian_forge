from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import order
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_create_order_without_type_fail(self):
    props = self.props.copy()
    del props['type']
    snippet = self.res_template.freeze(properties=props)
    self.assertRaisesRegex(exception.ResourceFailure, 'Property type not assigned', self._create_resource, 'foo', snippet, self.stack)