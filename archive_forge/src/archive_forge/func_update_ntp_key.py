from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def update_ntp_key(module, array):
    """Update NTP Symmetric Key"""
    if module.params['ntp_key'] == '' and (not getattr(list(array.get_arrays().items)[0], 'ntp_symmetric_key', None)):
        changed = False
    else:
        try:
            int(module.params['ntp_key'], 16)
            if len(module.params['ntp_key']) > 64:
                module.fail_json(msg='HEX string cannot be longer than 64 characters')
        except ValueError:
            if len(module.params['ntp_key']) > 20:
                module.fail_json(msg='ASCII string cannot be longer than 20 characters')
            if '#' in module.params['ntp_key']:
                module.fail_json(msg='ASCII string cannot contain # character')
            if not all((ord(c) < 128 for c in module.params['ntp_key'])):
                module.fail_json(msg='NTP key is non-ASCII')
        changed = True
        res = array.patch_arrays(array=Arrays(ntp_symmetric_key=module.params['ntp_key']))
        if res.status_code != 200:
            module.fail_json(msg='Failed to update NTP Symmetric Key. Error: {0}'.format(res.errors[0].message))
    module.exit_json(changed=changed)
    if len(module.params['ntp_key']) > 20:
        try:
            int(module.params['ntp_key'], 16)
        except ValueError:
            module.fail_json(msg='NTP key is not HEX')