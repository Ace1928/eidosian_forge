from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def probe_partnership(self, local_data, remote_data):
    modify_local, modify_remote = ({}, {})
    unsupported = []
    if self.link1:
        if local_data and local_data['link1'] != self.link1:
            unsupported.append('link1')
    if self.link2:
        if local_data and local_data['link2'] != self.link2:
            unsupported.append('link2')
    if self.remote_link1:
        if remote_data and remote_data['link1'] != self.remote_link1:
            unsupported.append('remote_link1')
    if self.remote_link2:
        if remote_data and remote_data['link2'] != self.remote_link2:
            unsupported.append('remote_link2')
    if self.type:
        if local_data and local_data['type'] != self.type or (remote_data and remote_data['type'] != self.type):
            unsupported.append('type')
    if unsupported:
        self.module.fail_json(msg='parameters {0} cannot be updated'.format(unsupported))
    if self.compressed:
        if local_data and local_data['compressed'] != self.compressed:
            modify_local['compressed'] = self.compressed
        if remote_data and remote_data['compressed'] != self.compressed:
            modify_remote['compressed'] = self.compressed
    if self.linkbandwidthmbits:
        if local_data and int(local_data['link_bandwidth_mbits']) != self.linkbandwidthmbits:
            modify_local['linkbandwidthmbits'] = self.linkbandwidthmbits
        if remote_data and int(remote_data['link_bandwidth_mbits']) != self.linkbandwidthmbits:
            modify_remote['linkbandwidthmbits'] = self.linkbandwidthmbits
    if self.backgroundcopyrate:
        if local_data and int(local_data['background_copy_rate']) != self.backgroundcopyrate:
            modify_local['backgroundcopyrate'] = self.backgroundcopyrate
        if remote_data and int(remote_data['background_copy_rate']) != self.backgroundcopyrate:
            modify_remote['backgroundcopyrate'] = self.backgroundcopyrate
    if self.remote_clusterip:
        if local_data and self.remote_clusterip != local_data['cluster_ip']:
            modify_local['clusterip'] = self.remote_clusterip
    return (modify_local, modify_remote)