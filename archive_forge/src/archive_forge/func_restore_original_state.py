from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def restore_original_state(self):
    """In case of failure restore, the changes we made."""
    for port, state in self.modified_ports.items():
        self.set_port_security_promiscuous([port], state)
    if self.deleted_session is not None:
        session = self.deleted_session
        config_version = self.dv_switch.config.configVersion
        s_spec = vim.dvs.VmwareDistributedVirtualSwitch.VspanConfigSpec(vspanSession=session, operation='add')
        c_spec = vim.dvs.VmwareDistributedVirtualSwitch.ConfigSpec(vspanConfigSpec=[s_spec], configVersion=config_version)
        task = self.dv_switch.ReconfigureDvs_Task(c_spec)
        try:
            wait_for_task(task)
        except Exception:
            self.restore_original_state()
            self.module.fail_json(msg=task.info.error.msg)