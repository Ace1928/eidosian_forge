from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import rax_argument_spec, rax_required_together, setup_rax_module
from ansible.module_utils.six import string_types
def rax_meta(module, address, name, server_id, meta):
    changed = False
    cs = pyrax.cloudservers
    if cs is None:
        module.fail_json(msg='Failed to instantiate client. This typically indicates an invalid region or an incorrectly capitalized region name.')
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
    if len(servers) > 1:
        module.fail_json(msg='Multiple servers found matching provided search parameters')
    elif not servers:
        module.fail_json(msg='Failed to find a server matching provided search parameters')
    for k, v in meta.items():
        if isinstance(v, list):
            meta[k] = ','.join(['%s' % i for i in v])
        elif isinstance(v, dict):
            meta[k] = json.dumps(v)
        elif not isinstance(v, string_types):
            meta[k] = '%s' % v
    server = servers[0]
    if server.metadata == meta:
        changed = False
    else:
        changed = True
        removed = set(server.metadata.keys()).difference(meta.keys())
        cs.servers.delete_meta(server, list(removed))
        cs.servers.set_meta(server, meta)
        server.get()
    module.exit_json(changed=changed, meta=server.metadata)