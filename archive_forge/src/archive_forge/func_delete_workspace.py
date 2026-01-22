from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.dict_transformations import _snake_to_camel, _camel_to_snake
def delete_workspace(self):
    try:
        self.log_analytics_client.workspaces.begin_delete(self.resource_group, self.name, force=self.force)
    except Exception as exc:
        self.fail('Error when deleting workspace {0} - {1}'.format(self.name, exc.message or str(exc)))