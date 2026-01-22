from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def verify_domain(self):
    response = self.get('domains/%s' % self.domain)
    status_code = response.status_code
    json = response.json
    if status_code not in (200, 404):
        self.module.fail_json(msg='Error getting domain [%(status_code)s: %(json)s]' % {'status_code': status_code, 'json': json})
    elif status_code == 404:
        self.module.fail_json(msg="No domain named '%s' found. Please create a domain first" % self.domain)