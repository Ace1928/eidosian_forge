import datetime
import json
from copy import deepcopy
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
from ansible_collections.community.aws.plugins.module_utils.opensearch import compare_domain_versions
from ansible_collections.community.aws.plugins.module_utils.opensearch import ensure_tags
from ansible_collections.community.aws.plugins.module_utils.opensearch import get_domain_config
from ansible_collections.community.aws.plugins.module_utils.opensearch import get_domain_status
from ansible_collections.community.aws.plugins.module_utils.opensearch import get_target_increment_version
from ansible_collections.community.aws.plugins.module_utils.opensearch import normalize_opensearch
from ansible_collections.community.aws.plugins.module_utils.opensearch import parse_version
from ansible_collections.community.aws.plugins.module_utils.opensearch import wait_for_domain_status
def set_vpc_options(module, current_domain_config, desired_domain_config, change_set):
    changed = False
    vpc_config = None
    if 'VPCOptions' in desired_domain_config:
        vpc_config = desired_domain_config['VPCOptions']
    vpc_opts = module.params.get('vpc_options')
    if vpc_opts is None:
        return changed
    vpc_subnets = vpc_opts.get('subnets')
    if vpc_subnets is not None:
        if vpc_config is None:
            vpc_config = {}
            desired_domain_config['VPCOptions'] = vpc_config
        if isinstance(vpc_subnets, string_types):
            vpc_subnets = [x.strip() for x in vpc_subnets.split(',')]
        vpc_config['SubnetIds'] = vpc_subnets
    vpc_security_groups = vpc_opts.get('security_groups')
    if vpc_security_groups is not None:
        if vpc_config is None:
            vpc_config = {}
            desired_domain_config['VPCOptions'] = vpc_config
        if isinstance(vpc_security_groups, string_types):
            vpc_security_groups = [x.strip() for x in vpc_security_groups.split(',')]
        vpc_config['SecurityGroupIds'] = vpc_security_groups
    if current_domain_config is not None:
        current_cluster_is_vpc = False
        desired_cluster_is_vpc = False
        if 'VPCOptions' in current_domain_config and 'SubnetIds' in current_domain_config['VPCOptions'] and (len(current_domain_config['VPCOptions']['SubnetIds']) > 0):
            current_cluster_is_vpc = True
        if 'VPCOptions' in desired_domain_config and 'SubnetIds' in desired_domain_config['VPCOptions'] and (len(desired_domain_config['VPCOptions']['SubnetIds']) > 0):
            desired_cluster_is_vpc = True
        if current_cluster_is_vpc != desired_cluster_is_vpc:
            change_set.append('VPCOptions changed between Internet and VPC')
            changed = True
        elif desired_cluster_is_vpc is False:
            pass
        else:
            if set(current_domain_config['VPCOptions']['SubnetIds']) != set(vpc_config['SubnetIds']):
                change_set.append(f'SubnetIds changed from {current_domain_config['VPCOptions']['SubnetIds']} to {vpc_config['SubnetIds']}')
                changed = True
            if set(current_domain_config['VPCOptions']['SecurityGroupIds']) != set(vpc_config['SecurityGroupIds']):
                change_set.append(f'SecurityGroup changed from {current_domain_config['VPCOptions']['SecurityGroupIds']} to {vpc_config['SecurityGroupIds']}')
                changed = True
    return changed