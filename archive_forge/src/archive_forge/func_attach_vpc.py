from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def attach_vpc(self, igw_id, vpc_id):
    try:
        self._connection.attach_internet_gateway(aws_retry=True, InternetGatewayId=igw_id, VpcId=vpc_id)
        waiter = get_waiter(self._connection, 'internet_gateway_attached')
        waiter.wait(InternetGatewayIds=[igw_id])
        self._results['changed'] = True
    except botocore.exceptions.WaiterError as e:
        self._module.fail_json_aws(e, msg='Failed to attach VPC.')