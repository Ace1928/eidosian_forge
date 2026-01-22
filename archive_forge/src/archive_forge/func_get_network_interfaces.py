from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def get_network_interfaces(connection, module, request_args):
    try:
        network_interfaces_result = connection.describe_network_interfaces(aws_retry=True, **request_args)
    except is_boto3_error_code('InvalidNetworkInterfaceID.NotFound'):
        module.exit_json(network_interfaces=[])
    except (ClientError, NoCredentialsError) as e:
        module.fail_json_aws(e)
    return network_interfaces_result