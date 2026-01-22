from unittest import mock
import yaml
from osc_lib import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine.resources.openstack.octavia import pool
from heat.tests import common
from heat.tests.openstack.octavia import inline_templates
from heat.tests import utils
def test_validate_source_ip_cookie_name(self):
    tmpl = yaml.safe_load(inline_templates.POOL_TEMPLATE)
    sp = tmpl['resources']['pool']['properties']['session_persistence']
    sp['type'] = 'SOURCE_IP'
    sp['cookie_name'] = 'cookie'
    self._create_stack(tmpl=yaml.safe_dump(tmpl))
    msg = _('Property cookie_name must NOT be specified when session_persistence type is set to SOURCE_IP.')
    self.assertRaisesRegex(exception.StackValidationFailed, msg, self.pool.validate)