from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
import re
import traceback
def inventory_vdos(module, vdocmd):
    rc, vdostatusout, err = module.run_command([vdocmd, 'status'])
    vdolist = []
    if rc == 2 and re.findall('vdoconf\\.yml does not exist', err, re.MULTILINE):
        return vdolist
    if rc != 0:
        module.fail_json(msg='Inventorying VDOs failed: %s' % vdostatusout, rc=rc, err=err)
    vdostatusyaml = yaml.safe_load(vdostatusout)
    if vdostatusyaml is None:
        return vdolist
    vdoyamls = vdostatusyaml['VDOs']
    if vdoyamls is not None:
        vdolist = list(vdoyamls.keys())
    return vdolist