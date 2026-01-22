from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import order
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_validate_certificate_with_profile_without_ca_id(self):
    props = self.props.copy()
    props['profile'] = 'cert'
    props['type'] = 'certificate'
    snippet = self.res_template.freeze(properties=props)
    res = self._create_resource('test', snippet, self.stack)
    msg = 'profile cannot be specified without ca_id.'
    self.assertRaisesRegex(exception.ResourcePropertyDependency, msg, res.validate)