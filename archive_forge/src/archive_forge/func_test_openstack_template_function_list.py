import yaml
from tempest.lib import exceptions
from heatclient.tests.functional.osc.v1 import base
def test_openstack_template_function_list(self):
    ret = self.openstack('orchestration template function list heat_template_version.2015-10-15')
    tmpl_functions = self.parser.listing(ret)
    self.assertTableStruct(tmpl_functions, ['Functions', 'Description'])