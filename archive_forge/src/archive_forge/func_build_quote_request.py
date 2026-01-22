from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def build_quote_request(self, source_we_info, dest_we_info, vol_dest_quote):
    quote = dict()
    quote['size'] = {'size': vol_dest_quote['size']['size'], 'unit': vol_dest_quote['size']['unit']}
    quote['name'] = self.parameters['destination_volume_name']
    quote['snapshotPolicyName'] = vol_dest_quote['snapshotPolicy']
    quote['enableDeduplication'] = vol_dest_quote['deduplication']
    quote['enableThinProvisioning'] = vol_dest_quote['thinProvisioning']
    quote['enableCompression'] = vol_dest_quote['compression']
    quote['verifyNameUniqueness'] = True
    quote['replicationFlow'] = True
    aggregate = self.get_aggregate_detail(source_we_info, vol_dest_quote['aggregateName'])
    if aggregate is None:
        self.module.fail_json(changed=False, msg='Error getting aggregate on source volume')
    if source_we_info['workingEnvironmentType'] != 'ON_PREM':
        if aggregate['providerVolumes'][0]['diskType'] == 'gp3' or aggregate['providerVolumes'][0]['diskType'] == 'io1' or aggregate['providerVolumes'][0]['diskType'] == 'io2':
            quote['iops'] = aggregate['providerVolumes'][0]['iops']
        if aggregate['providerVolumes'][0]['diskType'] == 'gp3':
            quote['throughput'] = aggregate['providerVolumes'][0]['throughput']
        quote['workingEnvironmentId'] = dest_we_info['publicId']
        quote['svmName'] = self.parameters['destination_svm_name']
    if self.parameters.get('capacity_tier') is not None:
        quote['capacityTier'] = self.parameters['capacity_tier']
    if self.parameters.get('provider_volume_type') is None:
        quote['providerVolumeType'] = vol_dest_quote['providerVolumeType']
    else:
        quote['providerVolumeType'] = self.parameters['provider_volume_type']
    return quote