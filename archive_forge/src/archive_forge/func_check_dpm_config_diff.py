from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def check_dpm_config_diff(self):
    """
        Check DRS configuration diff
        Returns: True if there is diff, else False

        """
    dpm_config = self.cluster.configurationEx.dpmConfigInfo
    change_message = None
    changes = False
    if dpm_config.enabled != self.enable_dpm:
        change_message = 'DPM enabled status changes'
        changes = True
        return (changes, change_message)
    elif self.enable_dpm:
        if dpm_config.hostPowerActionRate != self.host_power_action_rate or dpm_config.defaultDpmBehavior != self.default_dpm_behaviour:
            change_message = 'DPM Host Power Action Rate and/or default DPM behaviour change.'
            changes = True
            return (changes, change_message)
    return (changes, change_message)