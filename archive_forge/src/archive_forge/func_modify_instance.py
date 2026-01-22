from __future__ import (absolute_import, division, print_function)
import re
import time
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.general.plugins.module_utils.alicloud_ecs import (
def modify_instance(module, instance):
    state = module.params['state']
    name = module.params['instance_name']
    unique_suffix = module.params['unique_suffix']
    if not name:
        name = instance.name
    description = module.params['description']
    if not description:
        description = instance.description
    host_name = module.params['host_name']
    if unique_suffix and host_name:
        suffix = instance.host_name[-3:]
        host_name = host_name + suffix
    if not host_name:
        host_name = instance.host_name
    password = ''
    if state == 'restarted':
        password = module.params['password']
    setattr(instance, 'user_data', instance.describe_user_data())
    user_data = instance.user_data
    if state == 'stopped':
        user_data = module.params['user_data'].encode()
    try:
        return instance.modify(name=name, description=description, host_name=host_name, password=password, user_data=user_data)
    except Exception as e:
        module.fail_json(msg='Modify instance {0} attribute got an error: {1}'.format(instance.id, e))