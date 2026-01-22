import collections
from unittest import mock
from heat.common import exception
from heat.engine.clients.os import cinder as c_plugin
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_volume_type_handle_update_is_public(self):
    prop_diff = {'is_public': True, 'projects': []}
    self.patchobject(self.volume_type_access, 'list')
    volume_type_id = '927202df-1afb-497f-8368-9c2d2f26e5db'
    self.my_volume_type.resource_id = volume_type_id
    self.my_volume_type.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.volume_types.update.assert_called_once_with(volume_type_id, is_public=True)
    self.volume_type_access.list.assert_not_called()