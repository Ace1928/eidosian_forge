from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def is_snapshot_exists(self, old_name=None, force=False):
    old_name = old_name if old_name else self.name
    if self.volumegroup:
        data = self.lsvolumegroupsnapshot(old_name=old_name, force=force)
        self.parentuid = data.get('parent_uid')
    else:
        if self.lsv_data.get('snapshot_name') == old_name and (not force):
            return self.lsv_data
        cmdopts = {'filtervalue': 'snapshot_name={0}'.format(old_name)}
        result = self.restapi.svc_obj_info(cmd='lsvolumesnapshot', cmdopts=cmdopts, cmdargs=None)
        try:
            data = next(filter(lambda x: x['volume_group_name'] == '', result))
        except StopIteration:
            return {}
        else:
            self.lsv_data = data
            self.parentuid = data.get('parent_uid')
    return data