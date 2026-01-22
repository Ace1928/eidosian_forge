from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def nacl_entry_to_list(entry):
    elist = []
    elist.append(entry['rule_number'])
    if entry.get('protocol') in PROTOCOL_NAMES:
        elist.append(PROTOCOL_NAMES[entry['protocol']])
    else:
        elist.append(entry.get('protocol'))
    elist.append(entry['rule_action'])
    if entry.get('cidr_block'):
        elist.append(entry['cidr_block'])
    elif entry.get('ipv6_cidr_block'):
        elist.append(entry['ipv6_cidr_block'])
    else:
        elist.append(None)
    elist = elist + [None, None, None, None]
    if entry['protocol'] in ('1', '58'):
        elist[4] = entry.get('icmp_type_code', {}).get('type')
        elist[5] = entry.get('icmp_type_code', {}).get('code')
    if entry['protocol'] not in ('1', '6', '17', '58'):
        elist[6] = 0
        elist[7] = 65535
    elif 'port_range' in entry:
        elist[6] = entry['port_range']['from']
        elist[7] = entry['port_range']['to']
    return elist