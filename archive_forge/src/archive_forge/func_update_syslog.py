from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def update_syslog(module, array):
    """Update Syslog Server"""
    changed = False
    syslog_config = list(array.get_syslog_servers(names=[module.params['name']]).items)[0]
    noport_address = module.params['protocol'] + '://' + module.params['address']
    if module.params['port']:
        full_address = noport_address + ':' + module.params['port']
    else:
        full_address = noport_address
    if full_address != syslog_config.uri:
        changed = True
        res = array.patch_syslog_servers(names=[module.params['name']], syslog_server=SyslogServer(uri=full_address))
        if res.status_code != 200:
            module.fail_json(msg='Updating syslog server {0} failed. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)