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
def test_validate_hot_parameter_type(self):
    t = template_format.parse('\n            heat_template_version: 2013-05-23\n            parameters:\n              param1:\n                type: string\n              param2:\n                type: number\n              param3:\n                type: json\n              param4:\n                type: comma_delimited_list\n              param5:\n                type: boolean\n            ')
    res = dict(self.engine.validate_template(self.ctx, t, {}))
    parameters = res['Parameters']
    self.assertEqual('String', parameters['param1']['Type'])
    self.assertEqual('Number', parameters['param2']['Type'])
    self.assertEqual('Json', parameters['param3']['Type'])
    self.assertEqual('CommaDelimitedList', parameters['param4']['Type'])
    self.assertEqual('Boolean', parameters['param5']['Type'])