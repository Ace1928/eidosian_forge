import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import add_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
@staticmethod
def build_block_device_mapping(device_mapping):
    block_device_mapping = []
    for device in device_mapping:
        device = {k: v for k, v in device.items() if v is not None}
        device['Ebs'] = {}
        rename_item_if_exists(device, 'delete_on_termination', 'DeleteOnTermination', 'Ebs')
        rename_item_if_exists(device, 'device_name', 'DeviceName')
        rename_item_if_exists(device, 'encrypted', 'Encrypted', 'Ebs')
        rename_item_if_exists(device, 'iops', 'Iops', 'Ebs')
        rename_item_if_exists(device, 'no_device', 'NoDevice')
        rename_item_if_exists(device, 'size', 'VolumeSize', 'Ebs', attribute_type=int)
        rename_item_if_exists(device, 'snapshot_id', 'SnapshotId', 'Ebs')
        rename_item_if_exists(device, 'virtual_name', 'VirtualName')
        rename_item_if_exists(device, 'volume_size', 'VolumeSize', 'Ebs', attribute_type=int)
        rename_item_if_exists(device, 'volume_type', 'VolumeType', 'Ebs')
        if 'NoDevice' in device:
            if device['NoDevice'] is True:
                device['NoDevice'] = ''
            else:
                del device['NoDevice']
        block_device_mapping.append(device)
    return block_device_mapping