from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_carrier_gateway_info(carrier_gateway):
    tags = boto3_tag_list_to_ansible_dict(carrier_gateway['Tags'])
    ignore_list = []
    carrier_gateway_info = {'CarrierGatewayId': carrier_gateway['CarrierGatewayId'], 'VpcId': carrier_gateway['VpcId'], 'Tags': tags}
    carrier_gateway_info = camel_dict_to_snake_dict(carrier_gateway_info, ignore_list=ignore_list)
    return carrier_gateway_info