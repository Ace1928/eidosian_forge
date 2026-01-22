from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def update_dns(module, blade):
    """Set DNS Settings"""
    changed = False
    current_dns = blade.dns.list_dns()
    if module.params['domain']:
        if current_dns.items[0].domain != module.params['domain']:
            changed = True
            if not module.check_mode:
                try:
                    blade.dns.update_dns(dns_settings=Dns(domain=module.params['domain']))
                except Exception:
                    module.fail_json(msg='Update of DNS domain failed')
    if module.params['nameservers']:
        if sorted(module.params['nameservers']) != sorted(current_dns.items[0].nameservers):
            changed = True
            if not module.check_mode:
                try:
                    blade.dns.update_dns(dns_settings=Dns(nameservers=module.params['nameservers']))
                except Exception:
                    module.fail_json(msg='Update of DNS nameservers failed')
    module.exit_json(changed=changed)