from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
def update_traffic_shaping_policy(self, spec, results):
    """
        Update the traffic shaping policy according to the parameters
        Args:
            spec: The vSwitch spec
            results: The results dict

        Returns: True if changes have been made, else false
        """
    if not self.params['traffic_shaping'] or not spec.policy.nicTeaming:
        return False
    ts_policy = spec.policy.shapingPolicy
    changed = False
    ts_enabled = self.params['traffic_shaping'].get('enabled')
    if not ts_enabled:
        if ts_policy.enabled:
            ts_policy.enabled = False
            changed = True
        return changed
    for value in ['average_bandwidth', 'peak_bandwidth', 'burst_size']:
        if not self.params['traffic_shaping'].get(value):
            self.module.fail_json(msg='traffic_shaping.%s is a required parameter if traffic_shaping is enabled.' % value)
    ts_average_bandwidth = self.params['traffic_shaping'].get('average_bandwidth') * 1000
    ts_peak_bandwidth = self.params['traffic_shaping'].get('peak_bandwidth') * 1000
    ts_burst_size = self.params['traffic_shaping'].get('burst_size') * 1024
    if not ts_policy.enabled:
        ts_policy.enabled = True
        changed = True
    if ts_policy.averageBandwidth != ts_average_bandwidth:
        results['traffic_shaping_avg_bandw'] = ts_average_bandwidth
        results['traffic_shaping_avg_bandw_previous'] = ts_policy.averageBandwidth
        ts_policy.averageBandwidth = ts_average_bandwidth
        changed = True
    if ts_policy.peakBandwidth != ts_peak_bandwidth:
        results['traffic_shaping_peak_bandw'] = ts_peak_bandwidth
        results['traffic_shaping_peak_bandw_previous'] = ts_policy.peakBandwidth
        ts_policy.peakBandwidth = ts_peak_bandwidth
        changed = True
    if ts_policy.burstSize != ts_burst_size:
        results['traffic_shaping_burst'] = ts_burst_size
        results['traffic_shaping_burst_previous'] = ts_policy.burstSize
        ts_policy.burstSize = ts_burst_size
        changed = True
    return changed