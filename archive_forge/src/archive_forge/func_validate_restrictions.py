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
def validate_restrictions(self, config, restrictions, purge_restrictions=False):
    try:
        if restrictions is None:
            if purge_restrictions:
                return None
            else:
                return config
        self.validate_required_key('geo_restriction', 'restrictions.geo_restriction', restrictions)
        geo_restriction = restrictions.get('geo_restriction')
        self.validate_required_key('restriction_type', 'restrictions.geo_restriction.restriction_type', geo_restriction)
        existing_restrictions = config.get('geo_restriction', {}).get(geo_restriction['restriction_type'], {}).get('items', [])
        geo_restriction_items = geo_restriction.get('items')
        if not purge_restrictions:
            geo_restriction_items.extend([rest for rest in existing_restrictions if rest not in geo_restriction_items])
        valid_restrictions = ansible_list_to_cloudfront_list(geo_restriction_items)
        valid_restrictions['restriction_type'] = geo_restriction.get('restriction_type')
        return {'geo_restriction': valid_restrictions}
    except Exception as e:
        self.module.fail_json_aws(e, msg='Error validating restrictions')