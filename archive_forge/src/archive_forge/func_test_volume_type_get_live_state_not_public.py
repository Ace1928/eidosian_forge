import collections
from unittest import mock
from heat.common import exception
from heat.engine.clients.os import cinder as c_plugin
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_volume_type_get_live_state_not_public(self):
    self.my_volume_type.resource_id = '1234'
    volume_type = mock.Mock()
    volume_type.to_dict.return_value = {'name': 'test', 'is_public': False, 'description': 'test1', 'metadata': {'one': 'two'}}
    self.volume_types.get.return_value = volume_type
    volume_type.get_keys.return_value = {'one': 'two'}
    volume_type_access = mock.MagicMock()

    class Access(object):

        def __init__(self, idx, project, info):
            self.volumetype_id = idx
            self.project_id = project
            self.to_dict = mock.Mock(return_value=info)
    volume_type_access.list.return_value = [Access('1234', '1', {'volumetype_id': '1234', 'project_id': '1'}), Access('1234', '2', {'volumetype_id': '1234', 'project_id': '2'})]
    self.cinderclient.volume_type_access = volume_type_access
    reality = self.my_volume_type.get_live_state(self.my_volume_type.properties)
    expected = {'name': 'test', 'is_public': False, 'description': 'test1', 'metadata': {'one': 'two'}, 'projects': ['1', '2']}
    self.assertEqual(expected, reality)
    volume_type_access.list.assert_called_once_with('1234')