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
def set_cognito_options(module, current_domain_config, desired_domain_config, change_set):
    changed = False
    cognito_config = desired_domain_config['CognitoOptions']
    cognito_opts = module.params.get('cognito_options')
    if cognito_opts is None:
        return changed
    if cognito_opts.get('enabled') is not None:
        cognito_config['Enabled'] = cognito_opts.get('enabled')
    if not cognito_config['Enabled']:
        desired_domain_config['CognitoOptions'] = {'Enabled': False}
    else:
        if cognito_opts.get('cognito_user_pool_id') is not None:
            cognito_config['UserPoolId'] = cognito_opts.get('cognito_user_pool_id')
        if cognito_opts.get('cognito_identity_pool_id') is not None:
            cognito_config['IdentityPoolId'] = cognito_opts.get('cognito_identity_pool_id')
        if cognito_opts.get('cognito_role_arn') is not None:
            cognito_config['RoleArn'] = cognito_opts.get('cognito_role_arn')
    if current_domain_config is not None and current_domain_config['CognitoOptions'] != cognito_config:
        change_set.append(f'CognitoOptions changed from {current_domain_config['CognitoOptions']} to {cognito_config}')
        changed = True
    return changed