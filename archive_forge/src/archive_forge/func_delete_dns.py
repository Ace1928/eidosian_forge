from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_dns(module, blade):
    """Delete DNS Settings"""
    changed = True
    if not module.check_mode:
        changed = False
        current_dns = blade.dns.list_dns()
        if current_dns.items[0].domain or current_dns.items[0].nameservers != []:
            try:
                blade.dns.update_dns(dns_settings=Dns(domain='', nameservers=[]))
                changed = True
            except Exception:
                module.fail_json(msg='Deletion of DNS settings failed')
    module.exit_json(changed=changed)