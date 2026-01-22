import itertools
import json
import re
from collections import namedtuple
from copy import deepcopy
from ipaddress import ip_network
from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.network import to_ipv6_subnet
from ansible.module_utils.common.network import to_subnet
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_id
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def get_ip_permissions_sort_key(rule):
    RULE_KEYS_ALL = {'ip_ranges', 'ipv6_ranges', 'prefix_list_ids', 'user_id_group_pairs'}
    for rule_key in RULE_KEYS_ALL:
        if rule.get(rule_key):
            rule.get(rule_key).sort(key=get_rule_sort_key)
    if rule.get('ip_ranges'):
        value = str(rule.get('ip_ranges')[0]['cidr_ip'])
        return f'ipv4:{value}'
    if rule.get('ipv6_ranges'):
        value = str(rule.get('ipv6_ranges')[0]['cidr_ipv6'])
        return f'ipv6:{value}'
    if rule.get('prefix_list_ids'):
        value = str(rule.get('prefix_list_ids')[0]['prefix_list_id'])
        return f'pl:{value}'
    if rule.get('user_id_group_pairs'):
        value = str(rule.get('user_id_group_pairs')[0].get('group_id', ''))
        return f'ugid:{value}'
    return None