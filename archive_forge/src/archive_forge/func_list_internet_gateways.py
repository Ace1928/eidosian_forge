from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def list_internet_gateways(connection, module):
    params = dict()
    params['Filters'] = ansible_dict_to_boto3_filter_list(module.params.get('filters'))
    convert_tags = module.params.get('convert_tags')
    if module.params.get('internet_gateway_ids'):
        params['InternetGatewayIds'] = module.params.get('internet_gateway_ids')
    try:
        all_internet_gateways = connection.describe_internet_gateways(aws_retry=True, **params)
    except is_boto3_error_code('InvalidInternetGatewayID.NotFound'):
        module.fail_json('InternetGateway not found')
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, 'Unable to describe internet gateways')
    return [get_internet_gateway_info(igw, convert_tags) for igw in all_internet_gateways['InternetGateways']]