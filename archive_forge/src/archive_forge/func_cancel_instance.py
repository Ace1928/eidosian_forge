from __future__ import absolute_import, division, print_function
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
def cancel_instance(module):
    canceled = True
    if module.params.get('instance_id') is None and (module.params.get('tags') or module.params.get('hostname') or module.params.get('domain')):
        tags = module.params.get('tags')
        if isinstance(tags, string_types):
            tags = [module.params.get('tags')]
        instances = vsManager.list_instances(tags=tags, hostname=module.params.get('hostname'), domain=module.params.get('domain'))
        for instance in instances:
            try:
                vsManager.cancel_instance(instance['id'])
            except Exception:
                canceled = False
    elif module.params.get('instance_id') and module.params.get('instance_id') != 0:
        try:
            vsManager.cancel_instance(instance['id'])
        except Exception:
            canceled = False
    else:
        return (False, None)
    return (canceled, None)