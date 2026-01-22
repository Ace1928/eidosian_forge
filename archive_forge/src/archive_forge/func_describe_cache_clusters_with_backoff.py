from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_info
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.exponential_backoff()
def describe_cache_clusters_with_backoff(client, cluster_id=None):
    paginator = client.get_paginator('describe_cache_clusters')
    params = dict(ShowCacheNodeInfo=True)
    if cluster_id:
        params['CacheClusterId'] = cluster_id
    try:
        response = paginator.paginate(**params).build_full_result()
    except is_boto3_error_code('CacheClusterNotFound'):
        return []
    return response['CacheClusters']