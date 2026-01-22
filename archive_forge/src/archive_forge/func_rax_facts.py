from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import (rax_argument_spec,
def rax_facts(module, address, name, server_id):
    changed = False
    cs = pyrax.cloudservers
    if cs is None:
        module.fail_json(msg='Failed to instantiate client. This typically indicates an invalid region or an incorrectly capitalized region name.')
    ansible_facts = {}
    search_opts = {}
    if name:
        search_opts = dict(name='^%s$' % name)
        try:
            servers = cs.servers.list(search_opts=search_opts)
        except Exception as e:
            module.fail_json(msg='%s' % e.message)
    elif address:
        servers = []
        try:
            for server in cs.servers.list():
                for addresses in server.networks.values():
                    if address in addresses:
                        servers.append(server)
                        break
        except Exception as e:
            module.fail_json(msg='%s' % e.message)
    elif server_id:
        servers = []
        try:
            servers.append(cs.servers.get(server_id))
        except Exception as e:
            pass
    servers[:] = [server for server in servers if server.status != 'DELETED']
    if len(servers) > 1:
        module.fail_json(msg='Multiple servers found matching provided search parameters')
    elif len(servers) == 1:
        ansible_facts = rax_to_dict(servers[0], 'server')
    module.exit_json(changed=changed, ansible_facts=ansible_facts)