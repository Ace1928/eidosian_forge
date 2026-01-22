from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def gather_scsi_device_info(self):
    """
        Function to gather information about SCSI target devices

        """
    scsi_tgt_info = {}
    target_lun_uuid = {}
    scsilun_canonical = {}
    target_id = self.module.params['target_id']
    for host in self.hosts:
        for scsilun in host.config.storageDevice.scsiLun:
            scsilun_canonical[scsilun.key] = scsilun.canonicalName
        for target in host.config.storageDevice.scsiTopology.adapter[0].target:
            for lun in target.lun:
                target_lun_uuid[target.target] = lun.scsiLun
        scsi_tgt_info[host.name] = dict(scsilun_canonical=scsilun_canonical, target_lun_uuid=target_lun_uuid)
    if target_id is not None and self.esxi_hostname is not None:
        canonical = ''
        temp_lun_data = scsi_tgt_info[self.esxi_hostname]['target_lun_uuid']
        if self.esxi_hostname in scsi_tgt_info and target_id in temp_lun_data:
            temp_scsi_data = scsi_tgt_info[self.esxi_hostname]['scsilun_canonical']
            temp_target = temp_lun_data[target_id]
            canonical = temp_scsi_data[temp_target]
        self.module.exit_json(changed=False, canonical=canonical)
    self.module.exit_json(changed=False, scsi_tgt_info=scsi_tgt_info)