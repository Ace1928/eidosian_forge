import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
@ddt.data({'name': 'group2', 'desc': None, 'add': None, 'remove': None}, {'name': None, 'desc': 'group2 desc', 'add': None, 'remove': None}, {'name': None, 'desc': None, 'add': 'uuid1,uuid2', 'remove': None}, {'name': None, 'desc': None, 'add': None, 'remove': 'uuid3,uuid4'})
@ddt.unpack
def test_update_group_name(self, name, desc, add, remove):
    v = cs.groups.list()[0]
    expected = {'group': {'name': name, 'description': desc, 'add_volumes': add, 'remove_volumes': remove}}
    grp = v.update(name=name, description=desc, add_volumes=add, remove_volumes=remove)
    cs.assert_called('PUT', '/groups/1234', body=expected)
    self._assert_request_id(grp)
    grp = cs.groups.update('1234', name=name, description=desc, add_volumes=add, remove_volumes=remove)
    cs.assert_called('PUT', '/groups/1234', body=expected)
    self._assert_request_id(grp)
    grp = cs.groups.update(v, name=name, description=desc, add_volumes=add, remove_volumes=remove)
    cs.assert_called('PUT', '/groups/1234', body=expected)
    self._assert_request_id(grp)