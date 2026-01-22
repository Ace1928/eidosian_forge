from openstackclient.image.v2 import metadef_objects
from openstackclient.tests.unit.image.v2 import fakes
def test_object_set_no_options(self):
    arglist = [self._metadef_namespace.namespace, self._metadef_objects.name]
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.assertIsNone(result)