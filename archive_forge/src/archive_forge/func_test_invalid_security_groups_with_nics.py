from unittest import mock
from oslo_messaging.rpc import dispatcher
import webob
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.common import urlfetch
from heat.engine.clients.os import glance
from heat.engine import environment
from heat.engine.hot import template as hot_tmpl
from heat.engine import resources
from heat.engine import service
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
@mock.patch('heat.engine.clients.os.nova.NovaClientPlugin.client')
def test_invalid_security_groups_with_nics(self, mock_create):
    t = template_format.parse(test_template_invalid_secgroups)
    template = tmpl.Template(t, env=environment.Environment({'KeyName': 'test'}))
    stack = parser.Stack(self.ctx, 'test_stack', template)
    self._mock_get_image_id_success('image_id')
    mock_create.return_value = self.fc
    resource = stack['Instance']
    self.assertRaises(exception.ResourcePropertyConflict, resource.validate)