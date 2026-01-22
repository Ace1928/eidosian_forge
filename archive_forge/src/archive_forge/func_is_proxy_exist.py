from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def is_proxy_exist(self, obj_name):
    obj = {}
    result = self.restapi.svc_obj_info(cmd='lssystemsupportcenter', cmdopts=None, cmdargs=[obj_name])
    if isinstance(result, list):
        for d in result:
            obj.update(d)
    else:
        obj = result
    return obj