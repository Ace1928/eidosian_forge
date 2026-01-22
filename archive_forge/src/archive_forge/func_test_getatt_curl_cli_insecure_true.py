import datetime
from unittest import mock
import uuid
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from heat.common import identifier
from heat.common import template_format
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os import swift as swift_plugin
from heat.engine import environment
from heat.engine import resource
from heat.engine.resources.openstack.heat import wait_condition_handle as h_wch
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.objects import resource as resource_objects
from heat.tests import common
from heat.tests import utils
def test_getatt_curl_cli_insecure_true(self):
    self.patchobject(heat_plugin.HeatClientPlugin, 'get_heat_url', return_value='foo/%s' % self.tenant_id)
    self.patchobject(heat_plugin.HeatClientPlugin, 'get_insecure_option', return_value=True)
    handle = self._create_heat_handle()
    expected = "curl --insecure -i -X POST -H 'X-Auth-Token: adomainusertoken' -H 'Content-Type: application/json' -H 'Accept: application/json' foo/aprojectid/stacks/test_stack/%s/resources/wait_handle/signal" % self.stack_id
    self.assertEqual(expected, handle.FnGetAtt('curl_cli'))