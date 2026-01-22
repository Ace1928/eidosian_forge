from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import order
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_validate_non_certificate_order(self):
    props = self.props.copy()
    del props['bit_length']
    del props['algorithm']
    snippet = self.res_template.freeze(properties=props)
    res = self._create_resource('test', snippet, self.stack)
    msg = 'Properties algorithm and bit_length are required for key type of order.'
    self.assertRaisesRegex(exception.StackValidationFailed, msg, res.validate)