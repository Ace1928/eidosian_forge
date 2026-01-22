from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def describe_gateways(self, ip_address):
    response = self.ec2.describe_customer_gateways(DryRun=False, Filters=[{'Name': 'state', 'Values': ['available']}, {'Name': 'ip-address', 'Values': [ip_address]}])
    return response