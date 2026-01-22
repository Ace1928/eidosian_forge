from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
@AWSRetry.jittered_backoff()
def describe_eni_with_backoff(ec2, module, device_id):
    try:
        return ec2.describe_network_interfaces(NetworkInterfaceIds=[device_id])
    except is_boto3_error_code('InvalidNetworkInterfaceID.NotFound') as e:
        module.fail_json_aws(e, msg="Couldn't get list of network interfaces.")