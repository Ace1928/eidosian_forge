from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def configure_vsan(self):
    """
        Manage VSAN configuration

        """
    changed, result = (False, None)
    if self.check_vsan_config_diff():
        if not self.module.check_mode:
            vSanSpec = vim.vsan.ReconfigSpec(modify=True)
            vSanSpec.vsanClusterConfig = vim.vsan.cluster.ConfigInfo(enabled=self.enable_vsan)
            vSanSpec.vsanClusterConfig.defaultConfig = vim.vsan.cluster.ConfigInfo.HostDefaultInfo(autoClaimStorage=self.params.get('vsan_auto_claim_storage'))
            if self.advanced_options is not None:
                vSanSpec.extendedConfig = vim.vsan.VsanExtendedConfig()
                if self.advanced_options['automatic_rebalance'] is not None:
                    vSanSpec.extendedConfig.proactiveRebalanceInfo = vim.vsan.ProactiveRebalanceInfo(enabled=self.advanced_options['automatic_rebalance'])
                if self.advanced_options['disable_site_read_locality'] is not None:
                    vSanSpec.extendedConfig.disableSiteReadLocality = self.advanced_options['disable_site_read_locality']
                if self.advanced_options['large_cluster_support'] is not None:
                    vSanSpec.extendedConfig.largeScaleClusterSupport = self.advanced_options['large_cluster_support']
                if self.advanced_options['object_repair_timer'] is not None:
                    vSanSpec.extendedConfig.objectRepairTimer = self.advanced_options['object_repair_timer']
                if self.advanced_options['thin_swap'] is not None:
                    vSanSpec.extendedConfig.enableCustomizedSwapObject = self.advanced_options['thin_swap']
            try:
                task = self.vsanClusterConfigSystem.VsanClusterReconfig(self.cluster, vSanSpec)
                changed, result = wait_for_task(vim.Task(task._moId, self.si._stub))
            except vmodl.RuntimeFault as runtime_fault:
                self.module.fail_json(msg=to_native(runtime_fault.msg))
            except vmodl.MethodFault as method_fault:
                self.module.fail_json(msg=to_native(method_fault.msg))
            except TaskError as task_e:
                self.module.fail_json(msg=to_native(task_e))
            except Exception as generic_exc:
                self.module.fail_json(msg='Failed to update cluster due to generic exception %s' % to_native(generic_exc))
        else:
            changed = True
    self.module.exit_json(changed=changed, result=result)