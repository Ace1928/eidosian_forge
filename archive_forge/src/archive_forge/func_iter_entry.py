import re
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.plugin_utils.inventory import AWSInventoryBase
def iter_entry(self, hosts, hostnames, allow_duplicated_hosts=False, hostvars_prefix=None, hostvars_suffix=None, use_contrib_script_compatible_ec2_tag_keys=False):
    for host in hosts:
        if allow_duplicated_hosts:
            hostname_list = self._get_all_hostnames(host, hostnames)
        else:
            hostname_list = [self._get_preferred_hostname(host, hostnames)]
        if not hostname_list or hostname_list[0] is None:
            continue
        host_vars = _prepare_host_vars(host, hostvars_prefix, hostvars_suffix, use_contrib_script_compatible_ec2_tag_keys)
        for name in hostname_list:
            yield (to_text(name), host_vars)