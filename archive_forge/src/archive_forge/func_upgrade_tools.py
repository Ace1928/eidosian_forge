from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task
from ansible.module_utils._text import to_native
def upgrade_tools(self, vm):
    result = {'failed': False, 'changed': False, 'msg': ''}
    if vm.guest.toolsStatus == 'toolsOk':
        result.update(changed=False, msg='VMware tools is already up to date')
        return result
    elif vm.summary.runtime.powerState != 'poweredOn':
        result.update(failed=True, msg='VM must be powered on to upgrade tools')
        return result
    elif vm.guest.toolsStatus in ['toolsNotRunning', 'toolsNotInstalled']:
        result.update(failed=True, msg='VMware tools is either not running or not installed')
        return result
    elif vm.guest.toolsStatus == 'toolsOld':
        try:
            force = self.module.params.get('force_upgrade')
            installer_options = self.module.params.get('installer_options')
            if force or vm.guest.guestFamily in ['linuxGuest', 'windowsGuest']:
                if installer_options is not None:
                    task = vm.UpgradeTools(installer_options)
                else:
                    task = vm.UpgradeTools()
                changed, err_msg = wait_for_task(task)
                result.update(changed=changed, msg=to_native(err_msg))
            else:
                result.update(msg='Guest Operating System is other than Linux and Windows.')
            return result
        except Exception as exc:
            result.update(failed=True, msg='Error while upgrading VMware tools %s' % to_native(exc))
            return result
    else:
        result.update(failed=True, msg='VMware tools could not be upgraded')
        return result