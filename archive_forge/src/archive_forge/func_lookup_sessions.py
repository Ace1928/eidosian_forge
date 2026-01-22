from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.consul import (
def lookup_sessions(module, consul_module):
    datacenter = module.params.get('datacenter')
    state = module.params.get('state')
    try:
        if state == 'list':
            sessions_list = list_sessions(consul_module, datacenter)
            if sessions_list and len(sessions_list) >= 2:
                sessions_list = sessions_list[1]
            module.exit_json(changed=True, sessions=sessions_list)
        elif state == 'node':
            node = module.params.get('node')
            sessions = list_sessions_for_node(consul_module, node, datacenter)
            module.exit_json(changed=True, node=node, sessions=sessions)
        elif state == 'info':
            session_id = module.params.get('id')
            session_by_id = get_session_info(consul_module, session_id, datacenter)
            module.exit_json(changed=True, session_id=session_id, sessions=session_by_id)
    except Exception as e:
        module.fail_json(msg='Could not retrieve session info %s' % e)