from time import sleep
from time import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.network import to_subnet
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def wait_for_vpc_attribute(connection, module, vpc_id, attribute, expected_value):
    if expected_value is None:
        return
    if module.check_mode:
        return
    start_time = time()
    updated = False
    while time() < start_time + 300:
        current_value = connection.describe_vpc_attribute(Attribute=attribute, VpcId=vpc_id, aws_retry=True)[f'{attribute[0].upper()}{attribute[1:]}']['Value']
        if current_value != expected_value:
            sleep(3)
        else:
            updated = True
            break
    if not updated:
        module.fail_json(msg=f'Failed to wait for {attribute} to be updated')