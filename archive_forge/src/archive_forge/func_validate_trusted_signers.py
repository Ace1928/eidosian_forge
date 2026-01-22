import datetime
import re
from collections import OrderedDict
from ansible.module_utils._text import to_native
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.cloudfront_facts import CloudFrontFactsServiceManager
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def validate_trusted_signers(self, config, trusted_signers, cache_behavior):
    try:
        if trusted_signers is None:
            trusted_signers = {}
        if 'items' in trusted_signers:
            valid_trusted_signers = ansible_list_to_cloudfront_list(trusted_signers.get('items'))
        else:
            valid_trusted_signers = dict(quantity=config.get('quantity', 0))
            if 'items' in config:
                valid_trusted_signers = dict(items=config['items'])
        valid_trusted_signers['enabled'] = trusted_signers.get('enabled', config.get('enabled', self.__default_trusted_signers_enabled))
        cache_behavior['trusted_signers'] = valid_trusted_signers
        return cache_behavior
    except Exception as e:
        self.module.fail_json_aws(e, msg='Error validating trusted signers')