from openstackclient.image.v2 import metadef_namespaces
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
def test_namespace_show_no_options(self):
    arglist = [self._metadef_namespace.namespace]
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.expected_columns, columns)
    self.assertEqual(self.expected_data, data)