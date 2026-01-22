from openstackclient.image.v2 import metadef_objects
from openstackclient.tests.unit.image.v2 import fakes
def test_metadef_objects_list(self):
    arglist = [self._metadef_namespace.namespace]
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    self.assertEqual(getattr(self.datalist[0], 'name'), next(data)[0])