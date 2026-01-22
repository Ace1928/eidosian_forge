from __future__ import absolute_import, division, print_function
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.parsing import (
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible.module_utils.basic import AnsibleModule
def update_host_access_policies(module, current, patches):
    wanted = module.params
    if wanted['host_access_policies'] is not None:
        current_haps = extract_current_haps(current)
        wanted_haps = get_wanted_haps(module)
        if wanted_haps != current_haps:
            patch = purefusion.VolumePatch(host_access_policies=purefusion.NullableString(','.join(wanted_haps)))
            patches.append(patch)