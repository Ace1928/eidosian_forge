import collections
from unittest import mock
from heat.common import exception
from heat.engine.clients.os import cinder as c_plugin
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_volume_type_update_projects(self):
    self.my_volume_type.resource_id = '8aeaa446459a4d3196bc573fc252800b'
    prop_diff = {'projects': ['id2', 'id3'], 'is_public': False}

    class Access(object):

        def __init__(self, idx, project):
            self.volume_type_id = idx
            self.project_id = project
            info = {'volume_type_id': idx, 'project_id': project}
            self.to_dict = mock.Mock(return_value=info)
    old_access = [Access(self.my_volume_type.resource_id, 'id1'), Access(self.my_volume_type.resource_id, 'id2')]
    self.patchobject(self.volume_type_access, 'list', return_value=old_access)
    self.patchobject(self.volume_type_access, 'remove_project_access')
    project = collections.namedtuple('Project', ['id'])
    self.project_list.return_value = project('id3')
    self.my_volume_type.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.volume_type_access.remove_project_access.assert_called_once_with(self.my_volume_type.resource_id, 'id1')
    self.project_list.assert_called_once_with('id3')
    self.volume_type_access.add_project_access.assert_called_once_with(self.my_volume_type.resource_id, 'id3')