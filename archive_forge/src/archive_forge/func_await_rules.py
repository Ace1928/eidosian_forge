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
def await_rules(group, desired_rules, purge, rule_key):
    for _i in range(tries):
        current_rules = set(sum([list(rule_from_group_permission(p)) for p in group[rule_key]], []))
        if purge and len(current_rules ^ set(desired_rules)) == 0:
            return group
        elif purge:
            conflicts = current_rules ^ set(desired_rules)
            for a, b in itertools.combinations(conflicts, 2):
                if rule_cmp(a, b):
                    conflicts.discard(a)
                    conflicts.discard(b)
            if not len(conflicts):
                return group
        elif current_rules.issuperset(desired_rules) and (not purge):
            return group
        sleep(10)
        group = get_security_groups_with_backoff(client, GroupIds=[group_id])['SecurityGroups'][0]
    module.warn(f'Ran out of time waiting for {group_id} {rule_key}. Current: {current_rules}, Desired: {desired_rules}')
    return group