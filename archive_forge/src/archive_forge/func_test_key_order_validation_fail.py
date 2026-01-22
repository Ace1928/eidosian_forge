from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import order
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_key_order_validation_fail(self):
    props = self.props.copy()
    props['pass_phrase'] = 'something'
    snippet = self.res_template.freeze(properties=props)
    res = self._create_resource('test', snippet, self.stack)
    msg = 'Unexpected properties: pass_phrase. Only these properties are allowed for key type of order: algorithm, bit_length, expiration, mode, name, payload_content_type.'
    self.assertRaisesRegex(exception.StackValidationFailed, msg, res.validate)