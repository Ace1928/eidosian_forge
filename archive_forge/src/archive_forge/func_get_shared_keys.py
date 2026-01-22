from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.dict_transformations import _camel_to_snake
def get_shared_keys(self):
    try:
        return self.log_analytics_client.shared_keys.get_shared_keys(self.resource_group, self.name).as_dict()
    except Exception as exc:
        self.fail('Error when getting shared key {0}'.format(exc.message or str(exc)))