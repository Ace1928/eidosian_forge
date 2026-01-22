from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import normalize_ec2_vpc_dhcp_config
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
def match_dhcp_options(client, module, new_config):
    """
    Returns a DhcpOptionsId if the module parameters match; else None
    Filter by tags, if any are specified
    """
    try:
        all_dhcp_options = client.describe_dhcp_options(aws_retry=True)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg='Unable to describe dhcp options')
    for dopts in all_dhcp_options['DhcpOptions']:
        if module.params['tags']:
            boto_tags = ansible_dict_to_boto3_tag_list(module.params['tags'])
            if dopts['DhcpConfigurations'] == new_config and dopts['Tags'] == boto_tags:
                return (True, dopts['DhcpOptionsId'])
        elif dopts['DhcpConfigurations'] == new_config:
            return (True, dopts['DhcpOptionsId'])
    return (False, None)