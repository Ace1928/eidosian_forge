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
def rule_from_group_permission(perm):
    """
    Returns a rule dict from an existing security group.

    When using a security group as a target all 3 fields (OwnerId, GroupId, and
    GroupName) need to exist in the target. This ensures consistency of the
    values that will be compared to desired_ingress or desired_egress
    in wait_for_rule_propagation().
    GroupId is preferred as it is more specific except when targeting 'amazon-'
    prefixed security groups (such as EC2 Classic ELBs).
    """

    def ports_from_permission(p):
        if 'FromPort' not in p and 'ToPort' not in p:
            return (None, None)
        return (int(perm['FromPort']), int(perm['ToPort']))
    for target_key, target_subkey, target_type in [('IpRanges', 'CidrIp', 'ipv4'), ('Ipv6Ranges', 'CidrIpv6', 'ipv6'), ('PrefixListIds', 'PrefixListId', 'ip_prefix')]:
        if target_key not in perm:
            continue
        for r in perm[target_key]:
            yield Rule(ports_from_permission(perm), to_text(perm['IpProtocol']), r[target_subkey], target_type, r.get('Description'))
    if 'UserIdGroupPairs' in perm and perm['UserIdGroupPairs']:
        for pair in perm['UserIdGroupPairs']:
            target = (pair.get('UserId', current_account_id), pair.get('GroupId', None), None)
            if pair.get('UserId', '').startswith('amazon-'):
                target = (pair.get('UserId', None), None, pair.get('GroupName', None))
            elif 'VpcPeeringConnectionId' not in pair and pair['UserId'] != current_account_id:
                pass
            elif 'VpcPeeringConnectionId' in pair:
                target = (pair.get('UserId', None), pair.get('GroupId', None), None)
            yield Rule(ports_from_permission(perm), to_text(perm['IpProtocol']), target, 'group', pair.get('Description'))