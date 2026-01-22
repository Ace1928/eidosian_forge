from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.ec2 import BaseEc2Manager
from ansible_collections.community.aws.plugins.module_utils.ec2 import Boto3Mixin
from ansible_collections.community.aws.plugins.module_utils.ec2 import Ec2WaiterFactory
def subnets_to_vpc(self, subnets, subnet_details=None):
    if not subnets:
        return None
    if subnet_details is None:
        subnet_details = self._describe_subnets(SubnetIds=list(subnets))
    vpcs = [s.get('VpcId') for s in subnet_details]
    if len(set(vpcs)) > 1:
        self.module.fail_json(msg='Attachment subnets may only be in one VPC, multiple VPCs found', vpcs=list(set(vpcs)), subnets=subnet_details)
    return vpcs[0]