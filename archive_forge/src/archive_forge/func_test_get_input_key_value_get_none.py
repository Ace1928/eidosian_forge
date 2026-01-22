from unittest import mock
from heat.common import exception
from heat.engine.resources.openstack.heat import structured_config as sc
from heat.engine import rsrc_defn
from heat.engine import software_config_io as swc_io
from heat.engine import stack as parser
from heat.engine import template
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import utils
def test_get_input_key_value_get_none(self):
    inputs = {'bar': 'baz', 'foo': 'foo2'}
    res = sc.StructuredDeployment.get_input_key_value('brz', inputs, False)
    self.assertIsNone(res)