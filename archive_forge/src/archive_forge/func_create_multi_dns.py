from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def create_multi_dns(module, array):
    """Create a DNS configuration"""
    changed = True
    if not module.check_mode:
        if module.params['service'] == 'file':
            if module.params['source']:
                res = array.post_dns(names=[module.params['name']], dns=flasharray.DnsPost(services=[module.params['service']], domain=module.params['domain'], nameservers=module.params['nameservers'], source=flasharray.ReferenceNoId(module.params['source'].lower())))
            else:
                res = array.post_dns(names=[module.params['name']], dns=flasharray.DnsPost(services=[module.params['service']], domain=module.params['domain'], nameservers=module.params['nameservers']))
        else:
            res = array.create_dns(names=[module.params['name']], services=[module.params['service']], domain=module.params['domain'], nameservers=module.params['nameservers'])
        if res.status_code != 200:
            module.fail_json(msg='Failed to create {0} DNS configuration {1}. Error: {2}'.format(module.params['service'], module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)