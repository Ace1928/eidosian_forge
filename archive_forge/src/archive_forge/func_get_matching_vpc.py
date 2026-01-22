from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def get_matching_vpc(self, vpc_id):
    """
        Returns the virtual private cloud found.
            Parameters:
                vpc_id (str): VPC ID
            Returns:
                vpc (dict): dict of vpc found, None if none found
        """
    try:
        vpcs = describe_vpcs_with_backoff(self._connection, VpcIds=[vpc_id])
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        if 'InvalidVpcID.NotFound' in str(e):
            self._module.fail_json(msg=f'VPC with Id {vpc_id} not found, aborting')
        self._module.fail_json_aws(e)
    vpc = None
    if len(vpcs) > 1:
        self._module.fail_json(msg=f'EC2 returned more than one VPC for {vpc_id}, aborting')
    elif vpcs:
        vpc = camel_dict_to_snake_dict(vpcs[0])
    return vpc