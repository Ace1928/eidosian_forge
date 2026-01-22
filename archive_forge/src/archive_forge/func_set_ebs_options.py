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
def set_ebs_options(module, current_domain_config, desired_domain_config, change_set):
    changed = False
    ebs_config = desired_domain_config['EBSOptions']
    ebs_opts = module.params.get('ebs_options')
    if ebs_opts is None:
        return changed
    if ebs_opts.get('ebs_enabled') is not None:
        ebs_config['EBSEnabled'] = ebs_opts.get('ebs_enabled')
    if not ebs_config['EBSEnabled']:
        desired_domain_config['EBSOptions'] = {'EBSEnabled': False}
    else:
        if ebs_opts.get('volume_type') is not None:
            ebs_config['VolumeType'] = ebs_opts.get('volume_type')
        if ebs_opts.get('volume_size') is not None:
            ebs_config['VolumeSize'] = ebs_opts.get('volume_size')
        if ebs_opts.get('iops') is not None:
            ebs_config['Iops'] = ebs_opts.get('iops')
    if current_domain_config is not None and current_domain_config['EBSOptions'] != ebs_config:
        change_set.append(f'EBSOptions changed from {current_domain_config['EBSOptions']} to {ebs_config}')
        changed = True
    return changed