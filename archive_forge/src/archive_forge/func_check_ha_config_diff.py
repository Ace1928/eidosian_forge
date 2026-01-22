from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def check_ha_config_diff(self):
    """
        Check HA configuration diff
        Returns: True if there is diff, else False

        """
    das_config = self.cluster.configurationEx.dasConfig
    if das_config.enabled != self.enable_ha:
        return True
    if self.enable_ha and (das_config.vmMonitoring != self.params.get('ha_vm_monitoring') or das_config.hostMonitoring != self.params.get('ha_host_monitoring') or das_config.admissionControlEnabled != self.ha_admission_control or (das_config.defaultVmSettings.restartPriority != self.params.get('ha_restart_priority')) or (das_config.defaultVmSettings.isolationResponse != self.host_isolation_response) or (das_config.defaultVmSettings.vmToolsMonitoringSettings.vmMonitoring != self.params.get('ha_vm_monitoring')) or (das_config.defaultVmSettings.vmToolsMonitoringSettings.failureInterval != self.params.get('ha_vm_failure_interval')) or (das_config.defaultVmSettings.vmToolsMonitoringSettings.minUpTime != self.params.get('ha_vm_min_up_time')) or (das_config.defaultVmSettings.vmToolsMonitoringSettings.maxFailures != self.params.get('ha_vm_max_failures')) or (das_config.defaultVmSettings.vmToolsMonitoringSettings.maxFailureWindow != self.params.get('ha_vm_max_failure_window')) or (das_config.defaultVmSettings.vmComponentProtectionSettings.vmStorageProtectionForAPD != self.params.get('apd_response')) or (das_config.defaultVmSettings.vmComponentProtectionSettings.vmStorageProtectionForPDL != self.params.get('pdl_response'))):
        return True
    if self.ha_admission_control:
        if self.params.get('slot_based_admission_control'):
            policy = self.params.get('slot_based_admission_control')
            if not isinstance(das_config.admissionControlPolicy, vim.cluster.FailoverLevelAdmissionControlPolicy) or das_config.admissionControlPolicy.failoverLevel != policy.get('failover_level'):
                return True
        elif self.params.get('reservation_based_admission_control'):
            policy = self.params.get('reservation_based_admission_control')
            auto_compute_percentages = policy.get('auto_compute_percentages')
            if not isinstance(das_config.admissionControlPolicy, vim.cluster.FailoverResourcesAdmissionControlPolicy) or das_config.admissionControlPolicy.autoComputePercentages != auto_compute_percentages or das_config.admissionControlPolicy.failoverLevel != policy.get('failover_level'):
                return True
            if not auto_compute_percentages:
                if das_config.admissionControlPolicy.cpuFailoverResourcesPercent != policy.get('cpu_failover_resources_percent') or das_config.admissionControlPolicy.memoryFailoverResourcesPercent != policy.get('memory_failover_resources_percent'):
                    return True
        elif self.params.get('failover_host_admission_control'):
            policy = self.params.get('failover_host_admission_control')
            if not isinstance(das_config.admissionControlPolicy, vim.cluster.FailoverHostAdmissionControlPolicy):
                return True
            das_config.admissionControlPolicy.failoverHosts.sort(key=lambda h: h.name)
            if das_config.admissionControlPolicy.failoverHosts != self.get_failover_hosts():
                return True
    if self.params.get('apd_response') != 'disabled' and self.params.get('apd_response') != 'warning':
        if das_config.defaultVmSettings.vmComponentProtectionSettings.vmTerminateDelayForAPDSec != self.params.get('apd_delay'):
            return True
        if das_config.defaultVmSettings.vmComponentProtectionSettings.vmReactionOnAPDCleared != self.params.get('apd_reaction'):
            return True
    if self.changed_advanced_settings:
        return True
    return False