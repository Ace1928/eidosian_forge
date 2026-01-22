import time
from ipaddress import ip_address
from ipaddress import ip_network
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def correct_ips(connection, ip_list, module, eni_id):
    eni = describe_eni(connection, module, eni_id)
    private_addresses = set()
    if 'PrivateIpAddresses' in eni:
        for ip in eni['PrivateIpAddresses']:
            private_addresses.add(ip['PrivateIpAddress'])
    ip_set = set(ip_list)
    return ip_set.issubset(private_addresses)