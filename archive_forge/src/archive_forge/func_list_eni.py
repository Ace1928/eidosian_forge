from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def list_eni(connection, module, request_args):
    network_interfaces_result = get_network_interfaces(connection, module, request_args)
    camel_network_interfaces = []
    for network_interface in network_interfaces_result['NetworkInterfaces']:
        network_interface['TagSet'] = boto3_tag_list_to_ansible_dict(network_interface['TagSet'])
        network_interface['Tags'] = network_interface['TagSet']
        if 'Name' in network_interface['Tags']:
            network_interface['Name'] = network_interface['Tags']['Name']
        network_interface['Id'] = network_interface['NetworkInterfaceId']
        camel_network_interfaces.append(camel_dict_to_snake_dict(network_interface, ignore_list=['Tags', 'TagSet']))
    return camel_network_interfaces