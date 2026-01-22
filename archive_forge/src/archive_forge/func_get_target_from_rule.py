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
def get_target_from_rule(module, client, rule, name, group, groups, vpc_id, tags):
    """
    Returns tuple of (target_type, target, group_created) after validating rule params.

    rule: Dict describing a rule.
    name: Name of the security group being managed.
    groups: Dict of all available security groups.

    AWS accepts an ip range or a security group as target of a rule. This
    function validate the rule specification and return either a non-None
    group_id or a non-None ip range.

    When using a security group as a target all 3 fields (OwnerId, GroupId, and
    GroupName) need to exist in the target. This ensures consistency of the
    values that will be compared to current_rules (from current_ingress and
    current_egress) in wait_for_rule_propagation().
    """
    try:
        if rule.get('group_id'):
            return _target_from_rule_with_group_id(rule, groups)
        if 'group_name' in rule:
            return _target_from_rule_with_group_name(client, rule, name, group, groups, vpc_id, tags, module.check_mode)
        if 'cidr_ip' in rule:
            return ('ipv4', validate_ip(module, rule['cidr_ip']), False)
        if 'cidr_ipv6' in rule:
            return ('ipv6', validate_ip(module, rule['cidr_ipv6']), False)
        if 'ip_prefix' in rule:
            return ('ip_prefix', rule['ip_prefix'], False)
    except SecurityGroupError as e:
        e.fail(module)
    module.fail_json(msg='Could not match target for rule', failed_rule=rule)