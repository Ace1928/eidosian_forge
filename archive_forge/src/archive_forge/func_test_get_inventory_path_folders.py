import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_inventory_path_folders(self):
    ObjectContent = collections.namedtuple('ObjectContent', ['propSet'])
    DynamicProperty = collections.namedtuple('Property', ['name', 'val'])
    obj1 = ObjectContent(propSet=[DynamicProperty(name='Datacenter', val='dc-1')])
    obj2 = ObjectContent(propSet=[DynamicProperty(name='Datacenter', val='folder-2')])
    obj3 = ObjectContent(propSet=[DynamicProperty(name='Datacenter', val='folder-1')])
    objects = ['foo', 'bar', obj1, obj2, obj3]
    result = mock.sentinel.objects
    result.objects = objects
    session = mock.Mock()
    session.vim.RetrievePropertiesEx = mock.Mock()
    session.vim.RetrievePropertiesEx.return_value = result
    entity = mock.Mock()
    inv_path = vim_util.get_inventory_path(session.vim, entity, 100)
    self.assertEqual('/folder-2/dc-1', inv_path)