from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import WapiModule
from ..module_utils.api import NIOS_NSGROUP
from ..module_utils.api import normalize_ib_spec
def ext_primaries_transform(module):
    if module.params['external_primaries']:
        for ext in module.params['external_primaries']:
            clean_tsig(ext)
    return module.params['external_primaries']