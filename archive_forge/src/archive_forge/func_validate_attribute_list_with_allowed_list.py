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
def validate_attribute_list_with_allowed_list(self, attribute_list, attribute_list_name, allowed_list):
    try:
        self.validate_is_list(attribute_list, attribute_list_name)
        if isinstance(allowed_list, list) and set(attribute_list) not in allowed_list or (isinstance(allowed_list, set) and (not set(allowed_list).issuperset(attribute_list))):
            attribute_list = ' '.join((str(a) for a in allowed_list))
            self.module.fail_json(msg=f'The attribute list {attribute_list_name} must be one of [{attribute_list}]')
    except Exception as e:
        self.module.fail_json_aws(e, msg='Error validating attribute list with allowed value list')