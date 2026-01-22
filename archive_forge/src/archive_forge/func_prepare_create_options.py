import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def prepare_create_options(module):
    """
    Return data structure for cluster create operation
    """
    c_params = {'ClusterName': module.params['name'], 'KafkaVersion': module.params['version'], 'ConfigurationInfo': {'Arn': module.params['configuration_arn'], 'Revision': module.params['configuration_revision']}, 'NumberOfBrokerNodes': module.params['nodes'], 'BrokerNodeGroupInfo': {'ClientSubnets': module.params['subnets'], 'InstanceType': module.params['instance_type']}}
    if module.params['security_groups'] and len(module.params['security_groups']) != 0:
        c_params['BrokerNodeGroupInfo']['SecurityGroups'] = module.params.get('security_groups')
    if module.params['ebs_volume_size']:
        c_params['BrokerNodeGroupInfo']['StorageInfo'] = {'EbsStorageInfo': {'VolumeSize': module.params.get('ebs_volume_size')}}
    if module.params['encryption']:
        c_params['EncryptionInfo'] = {}
        if module.params['encryption'].get('kms_key_id'):
            c_params['EncryptionInfo']['EncryptionAtRest'] = {'DataVolumeKMSKeyId': module.params['encryption']['kms_key_id']}
        c_params['EncryptionInfo']['EncryptionInTransit'] = {'ClientBroker': module.params['encryption']['in_transit'].get('client_broker', 'TLS'), 'InCluster': module.params['encryption']['in_transit'].get('in_cluster', True)}
    if module.params['authentication']:
        c_params['ClientAuthentication'] = {}
        if module.params['authentication'].get('sasl_scram') or module.params['authentication'].get('sasl_iam'):
            sasl = {}
            if module.params['authentication'].get('sasl_scram'):
                sasl['Scram'] = {'Enabled': True}
            if module.params['authentication'].get('sasl_iam'):
                sasl['Iam'] = {'Enabled': True}
            c_params['ClientAuthentication']['Sasl'] = sasl
        if module.params['authentication'].get('tls_ca_arn'):
            c_params['ClientAuthentication']['Tls'] = {'CertificateAuthorityArnList': module.params['authentication']['tls_ca_arn'], 'Enabled': True}
        if module.params['authentication'].get('unauthenticated'):
            c_params['ClientAuthentication'] = {'Unauthenticated': {'Enabled': True}}
    c_params.update(prepare_enhanced_monitoring_options(module))
    c_params.update(prepare_open_monitoring_options(module))
    c_params.update(prepare_logging_options(module))
    return c_params