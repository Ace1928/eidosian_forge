from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def probe_fc_partnership(self):
    probe_data = {}
    if self.linkbandwidthmbits and self.linkbandwidthmbits != self.partnership_data.get('link_bandwidth_mbits'):
        probe_data['linkbandwidthmbits'] = self.linkbandwidthmbits
    if self.backgroundcopyrate and self.backgroundcopyrate != self.partnership_data.get('background_copy_rate'):
        probe_data['backgroundcopyrate'] = self.backgroundcopyrate
    if self.pbrinuse and self.pbrinuse != self.partnership_data.get('pbr_in_use'):
        probe_data['pbrinuse'] = self.pbrinuse
    if self.start in {True, False}:
        probe_data['start'] = self.start
    if self.stop in {True, False}:
        probe_data['stop'] = self.stop
    return probe_data