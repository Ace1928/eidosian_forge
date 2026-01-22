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
def validate_allowed_methods(self, config, allowed_methods, cache_behavior):
    try:
        if allowed_methods is not None:
            self.validate_required_key('items', 'cache_behavior.allowed_methods.items[]', allowed_methods)
            temp_allowed_items = allowed_methods.get('items')
            self.validate_is_list(temp_allowed_items, 'cache_behavior.allowed_methods.items')
            self.validate_attribute_list_with_allowed_list(temp_allowed_items, 'cache_behavior.allowed_methods.items[]', self.__valid_methods_allowed_methods)
            cached_items = allowed_methods.get('cached_methods')
            if 'cached_methods' in allowed_methods:
                self.validate_is_list(cached_items, 'cache_behavior.allowed_methods.cached_methods')
                self.validate_attribute_list_with_allowed_list(cached_items, 'cache_behavior.allowed_items.cached_methods[]', self.__valid_methods_cached_methods)
            if 'allowed_methods' in config and set(config['allowed_methods']['items']) == set(temp_allowed_items):
                cache_behavior['allowed_methods'] = config['allowed_methods']
            else:
                cache_behavior['allowed_methods'] = ansible_list_to_cloudfront_list(temp_allowed_items)
            if cached_items and set(cached_items) == set(config.get('allowed_methods', {}).get('cached_methods', {}).get('items', [])):
                cache_behavior['allowed_methods']['cached_methods'] = config['allowed_methods']['cached_methods']
            else:
                cache_behavior['allowed_methods']['cached_methods'] = ansible_list_to_cloudfront_list(cached_items)
        elif 'allowed_methods' in config:
            cache_behavior['allowed_methods'] = config.get('allowed_methods')
        return cache_behavior
    except Exception as e:
        self.module.fail_json_aws(e, msg='Error validating allowed methods')