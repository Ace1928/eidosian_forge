from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
def sync_default_lb_vserver(client, module):
    d = get_default_lb_vserver(client, module)
    if module.params['lbvserver'] is not None:
        configured = ConfigProxy(actual=csvserver_lbvserver_binding(), client=client, readwrite_attrs=['name', 'lbvserver'], attribute_values_dict={'name': module.params['name'], 'lbvserver': module.params['lbvserver']})
        if not configured.has_equal_attributes(d):
            if d.name is not None:
                log('Deleting default lb vserver %s' % d.lbvserver)
                csvserver_lbvserver_binding.delete(client, d)
            log('Adding default lb vserver %s' % configured.lbvserver)
            configured.add()
    elif d.name is not None:
        log('Deleting default lb vserver %s' % d.lbvserver)
        csvserver_lbvserver_binding.delete(client, d)