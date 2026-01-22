import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import describe_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def get_eip_allocation_id_by_address(client, module, eip_address):
    """Release an EIP from your EIP Pool
    Args:
        client (botocore.client.EC2): Boto3 client
        module: AnsibleAWSModule class instance
        eip_address (str): The Elastic IP Address of the EIP.

    Basic Usage:
        >>> client = boto3.client('ec2')
        >>> module = AnsibleAWSModule(...)
        >>> eip_address = '52.87.29.36'
        >>> get_eip_allocation_id_by_address(client, module, eip_address)
        (
            'eipalloc-36014da3', ''
        )

    Returns:
        Tuple (str, str)
    """
    params = {'PublicIps': [eip_address]}
    allocation_id = None
    msg = ''
    try:
        allocations = client.describe_addresses(aws_retry=True, **params)['Addresses']
        if len(allocations) == 1:
            allocation = allocations[0]
        else:
            allocation = None
        if allocation:
            if allocation.get('Domain') != 'vpc':
                msg = f'EIP {eip_address} is a non-VPC EIP, please allocate a VPC scoped EIP'
            else:
                allocation_id = allocation.get('AllocationId')
    except is_boto3_error_code('InvalidAddress.Malformed'):
        module.fail_json(msg=f'EIP address {eip_address} is invalid.')
    except is_boto3_error_code('InvalidAddress.NotFound'):
        msg = f'EIP {eip_address} does not exist'
        allocation_id = None
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Unable to describe EIP')
    return (allocation_id, msg)