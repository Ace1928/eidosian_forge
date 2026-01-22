from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def modify_members_flow(module, client, members, result):
    debug = module.params['debug']
    force = module.params['force']
    max_time_ms = module.params['max_time_ms']
    diff = False
    modified_config = None
    config = None
    try:
        config = get_replicaset_config(client)
    except Exception as excep:
        module.fail_json(msg='Unable to get replicaset configuration {0}'.format(excep))
    if isinstance(members[0], str):
        diff = lists_are_different(members, get_member_names(client))
    elif isinstance(members[0], dict):
        diff = member_dicts_different(config, members)
    else:
        module.fail_json(msg='members must be either str or dict')
    if diff:
        if not module.check_mode:
            try:
                modified_config = modify_members(module, config, members)
                if debug:
                    result['config'] = str(config)
                    result['modified_config'] = str(modified_config)
                replicaset_reconfigure(module, client, modified_config, force, max_time_ms)
            except Exception as excep:
                module.fail_json(msg='Failed reconfiguring replicaset {0}, config doc {1}'.format(excep, modified_config))
        result['changed'] = True
        result['msg'] = 'replicaset reconfigured'
    else:
        result['changed'] = False
    return result