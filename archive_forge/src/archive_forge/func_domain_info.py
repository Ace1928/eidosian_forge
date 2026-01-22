from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils import deps
def domain_info(dnsimple_mod, req_obj):
    req_obj.url, req_obj.method = (req_obj.url + '/zones/' + dnsimple_mod.params['name'] + '/records?per_page=100', 'GET')
    return iterate_data(dnsimple_mod, req_obj)