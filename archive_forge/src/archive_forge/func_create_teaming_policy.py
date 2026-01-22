from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
def create_teaming_policy(self):
    """
        Create a NIC Teaming Policy
        Returns: NIC Teaming Policy object
        """
    if not all((option is None for option in [self.teaming_load_balancing, self.teaming_failure_detection, self.teaming_notify_switches, self.teaming_failback, self.teaming_failover_order_active, self.teaming_failover_order_standby])):
        teaming_policy = vim.host.NetworkPolicy.NicTeamingPolicy()
        teaming_policy.policy = self.teaming_load_balancing
        teaming_policy.reversePolicy = True
        teaming_policy.notifySwitches = self.teaming_notify_switches
        if self.teaming_failback is None:
            teaming_policy.rollingOrder = None
        else:
            teaming_policy.rollingOrder = not self.teaming_failback
        if self.teaming_failover_order_active is None and self.teaming_failover_order_standby is None:
            teaming_policy.nicOrder = None
        else:
            teaming_policy.nicOrder = self.create_nic_order_policy()
        if self.teaming_failure_detection is None:
            teaming_policy.failureCriteria = None
        else:
            teaming_policy.failureCriteria = self.create_nic_failure_policy()
        return teaming_policy
    return None