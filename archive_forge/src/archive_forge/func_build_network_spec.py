import time
import uuid
from collections import namedtuple
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.arn import validate_aws_arn
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.exceptions import AnsibleAWSError
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.tower import tower_callback_script
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def build_network_spec(params):
    """
    Returns list of interfaces [complex]
    Interface type: {
        'AssociatePublicIpAddress': True|False,
        'DeleteOnTermination': True|False,
        'Description': 'string',
        'DeviceIndex': 123,
        'Groups': [
            'string',
        ],
        'Ipv6AddressCount': 123,
        'Ipv6Addresses': [
            {
                'Ipv6Address': 'string'
            },
        ],
        'NetworkInterfaceId': 'string',
        'PrivateIpAddress': 'string',
        'PrivateIpAddresses': [
            {
                'Primary': True|False,
                'PrivateIpAddress': 'string'
            },
        ],
        'SecondaryPrivateIpAddressCount': 123,
        'SubnetId': 'string'
    },
    """
    interfaces = []
    network = params.get('network') or {}
    if not network.get('interfaces'):
        spec = {'DeviceIndex': 0}
        if network.get('assign_public_ip') is not None:
            spec['AssociatePublicIpAddress'] = network['assign_public_ip']
        if params.get('vpc_subnet_id'):
            spec['SubnetId'] = params['vpc_subnet_id']
        else:
            default_vpc = get_default_vpc()
            if default_vpc is None:
                module.fail_json(msg='No default subnet could be found - you must include a VPC subnet ID (vpc_subnet_id parameter) to create an instance')
            else:
                sub = get_default_subnet(default_vpc, availability_zone=module.params.get('availability_zone'))
                spec['SubnetId'] = sub['SubnetId']
        if network.get('private_ip_address'):
            spec['PrivateIpAddress'] = network['private_ip_address']
        if params.get('security_group') or params.get('security_groups'):
            groups = discover_security_groups(group=params.get('security_group'), groups=params.get('security_groups'), subnet_id=spec['SubnetId'])
            spec['Groups'] = groups
        if network.get('description') is not None:
            spec['Description'] = network['description']
        return [spec]
    for idx, interface_params in enumerate(network.get('interfaces', [])):
        spec = {'DeviceIndex': idx}
        if isinstance(interface_params, string_types):
            interface_params = {'id': interface_params}
        if interface_params.get('id') is not None:
            spec['NetworkInterfaceId'] = interface_params['id']
            interfaces.append(spec)
            continue
        spec['DeleteOnTermination'] = interface_params.get('delete_on_termination', True)
        if interface_params.get('ipv6_addresses'):
            spec['Ipv6Addresses'] = [{'Ipv6Address': a} for a in interface_params.get('ipv6_addresses', [])]
        if interface_params.get('private_ip_address'):
            spec['PrivateIpAddress'] = interface_params.get('private_ip_address')
        if interface_params.get('description'):
            spec['Description'] = interface_params.get('description')
        if interface_params.get('subnet_id', params.get('vpc_subnet_id')):
            spec['SubnetId'] = interface_params.get('subnet_id', params.get('vpc_subnet_id'))
        elif not spec.get('SubnetId') and (not interface_params['id']):
            raise ValueError(f'Failed to assign subnet to interface {interface_params}')
        interfaces.append(spec)
    return interfaces