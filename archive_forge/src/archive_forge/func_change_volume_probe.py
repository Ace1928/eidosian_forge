from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def change_volume_probe(self):
    is_update_required = False
    rcrelationship_data = self.get_existing_rc()
    if not rcrelationship_data:
        self.module.fail_json(msg="Relationship '%s' does not exists, relationship must exists before calling this module" % self.rname)
    if self.ismaster:
        if self.cvname == rcrelationship_data['master_change_vdisk_name']:
            self.log('Master change volume %s is already attached to the relationship', self.cvname)
        elif rcrelationship_data['master_change_vdisk_name'] != '':
            self.module.fail_json(msg='Master change volume %s is already attached to the relationship' % rcrelationship_data['master_change_vdisk_name'])
        else:
            is_update_required = True
    elif self.cvname == rcrelationship_data['aux_change_vdisk_name']:
        self.log('Aux change volume %s is already attached to the relationship', self.cvname)
    elif rcrelationship_data['aux_change_vdisk_name'] != '':
        self.module.fail_json(msg='Aux change volume %s is already attached to the relationship' % rcrelationship_data['aux_change_vdisk_name'])
    else:
        is_update_required = True
    return is_update_required