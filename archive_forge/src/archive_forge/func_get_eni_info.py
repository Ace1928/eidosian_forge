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
def get_eni_info(interface):
    private_addresses = []
    if 'PrivateIpAddresses' in interface:
        for ip in interface['PrivateIpAddresses']:
            private_addresses.append({'private_ip_address': ip['PrivateIpAddress'], 'primary_address': ip['Primary']})
    groups = {}
    if 'Groups' in interface:
        for group in interface['Groups']:
            groups[group['GroupId']] = group['GroupName']
    interface_info = {'id': interface.get('NetworkInterfaceId'), 'subnet_id': interface.get('SubnetId'), 'vpc_id': interface.get('VpcId'), 'description': interface.get('Description'), 'owner_id': interface.get('OwnerId'), 'status': interface.get('Status'), 'mac_address': interface.get('MacAddress'), 'private_ip_address': interface.get('PrivateIpAddress'), 'source_dest_check': interface.get('SourceDestCheck'), 'groups': groups, 'private_ip_addresses': private_addresses}
    if 'TagSet' in interface:
        tags = boto3_tag_list_to_ansible_dict(interface['TagSet'])
        if 'Name' in tags:
            interface_info['name'] = tags['Name']
        interface_info['tags'] = tags
    if 'Attachment' in interface:
        interface_info['attachment'] = {'attachment_id': interface['Attachment'].get('AttachmentId'), 'instance_id': interface['Attachment'].get('InstanceId'), 'device_index': interface['Attachment'].get('DeviceIndex'), 'status': interface['Attachment'].get('Status'), 'attach_time': interface['Attachment'].get('AttachTime'), 'delete_on_termination': interface['Attachment'].get('DeleteOnTermination')}
    return interface_info