from unittest import mock
from heatclient import exc
from heatclient.osc.v1 import template
from heatclient.tests.unit.osc.v1 import fakes
from heatclient.v1 import template_versions
def test_function_list(self):
    arglist = ['version1']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.template_versions.get.return_value = [self.tv1, self.tv2]
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(['Functions', 'Description'], columns)
    self.assertEqual([('func1', 'Function 1'), ('func2', 'Function 2')], list(data))