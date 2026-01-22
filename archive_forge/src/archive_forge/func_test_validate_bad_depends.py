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
def test_validate_bad_depends(self):
    test_template = '\n        heat_template_version: 2013-05-23\n\n        resources:\n          random_str:\n            type: OS::Heat::RandomString\n            depends_on: [{foo: bar}]\n        '
    t = template_format.parse(test_template)
    res = dict(self.engine.validate_template(self.ctx, t, {}))
    self.assertEqual({'Error': 'Resource random_str depends_on must be a list of strings'}, res)