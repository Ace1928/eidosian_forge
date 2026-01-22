from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.vyos import (
def config_to_dict(module):
    data = get_config(module)
    config = {'domain_search': [], 'name_server': []}
    for line in data.split('\n'):
        if line.startswith('set system host-name'):
            config['host_name'] = line[22:-1]
        elif line.startswith('set system domain-name'):
            config['domain_name'] = line[24:-1]
        elif line.startswith('set system domain-search domain'):
            config['domain_search'].append(line[33:-1])
        elif line.startswith('set system name-server'):
            config['name_server'].append(line[24:-1])
    return config