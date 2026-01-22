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
def test_build_derived_config_failure(self):
    props = {'input_values_validate': 'STRICT'}
    self.template['Resources']['deploy_mysql']['Properties'] = props
    self._stack_with_template(self.template)
    self.assertRaises(exception.UserParameterMissing, self.deployment._build_derived_config, 'CREATE', self.source, self.inputs[:1], {})