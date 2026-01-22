import copy
from unittest import mock
from openstackclient.api import object_store_v1 as object_store
from openstackclient.object.v1 import object as obj
from openstackclient.tests.unit.object.v1 import fakes as object_fakes
def test_object_list_objects_prefix(self, o_mock):
    o_mock.return_value = [copy.deepcopy(object_fakes.OBJECT_2)]
    arglist = ['--prefix', 'floppy', object_fakes.container_name_2]
    verifylist = [('prefix', 'floppy'), ('container', object_fakes.container_name_2)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'prefix': 'floppy'}
    o_mock.assert_called_with(container=object_fakes.container_name_2, **kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, tuple(data))