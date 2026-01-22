from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.scaleway import SCALEWAY_LOCATION, scaleway_argument_spec, Scaleway
from ansible.module_utils.basic import AnsibleModule
def ip_attributes_should_be_changed(api, target_ip, wished_ip):
    patch_payload = {}
    if target_ip['reverse'] != wished_ip['reverse']:
        patch_payload['reverse'] = wished_ip['reverse']
    if target_ip['server'] is None and wished_ip['server']:
        patch_payload['server'] = wished_ip['server']
    try:
        if target_ip['server']['id'] and wished_ip['server'] is None:
            patch_payload['server'] = wished_ip['server']
    except (TypeError, KeyError):
        pass
    try:
        if target_ip['server']['id'] != wished_ip['server']:
            patch_payload['server'] = wished_ip['server']
    except (TypeError, KeyError):
        pass
    return patch_payload