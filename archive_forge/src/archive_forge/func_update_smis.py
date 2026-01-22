from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def update_smis(module, array):
    """Update SMI-S features"""
    changed = smis_changed = False
    try:
        current = list(array.get_smi_s().items)[0]
    except Exception:
        module.fail_json(msg='Failed to get current SMI-S settings.')
    slp_enabled = current.slp_enabled
    wbem_enabled = current.wbem_https_enabled
    if slp_enabled != module.params['slp']:
        slp_enabled = module.params['slp']
        smis_changed = True
    if wbem_enabled != module.params['smis']:
        wbem_enabled = module.params['smis']
        smis_changed = True
    if smis_changed:
        smi_s = flasharray.Smis(slp_enabled=slp_enabled, wbem_https_enabled=wbem_enabled)
        changed = True
        if not module.check_mode:
            try:
                array.patch_smi_s(smi_s=smi_s)
            except Exception:
                module.fail_json(msg='Failed to change SMI-S settings.')
    module.exit_json(changed=changed)