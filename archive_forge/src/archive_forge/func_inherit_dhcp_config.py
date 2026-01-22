from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import normalize_ec2_vpc_dhcp_config
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
def inherit_dhcp_config(existing_config, new_config):
    """
    Compare two DhcpConfigurations lists and apply existing options to unset parameters

    If there's an existing option config and the new option is not set or it's none,
    inherit the existing config.
    The configs are unordered lists of dicts with non-unique keys, so we have to find
    the right list index for a given config option first.
    """
    changed = False
    for option in ['domain-name', 'domain-name-servers', 'ntp-servers', 'netbios-name-servers', 'netbios-node-type']:
        existing_index = find_opt_index(existing_config, option)
        new_index = find_opt_index(new_config, option)
        if existing_index is not None and new_index is None:
            new_config.append(existing_config[existing_index])
            changed = True
    return (changed, new_config)