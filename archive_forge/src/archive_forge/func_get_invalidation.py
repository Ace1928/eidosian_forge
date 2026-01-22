import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.cloudfront_facts import CloudFrontFactsServiceManager
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_invalidation(self, distribution_id, caller_reference):
    invalidations = self.__cloudfront_facts_mgr.list_invalidations(distribution_id=distribution_id)
    for invalidation in invalidations:
        invalidation_info = self.__cloudfront_facts_mgr.get_invalidation(distribution_id=distribution_id, id=invalidation['Id'])
        if invalidation_info.get('InvalidationBatch', {}).get('CallerReference') == caller_reference:
            invalidation_info.pop('ResponseMetadata', None)
            return invalidation_info
    return {}