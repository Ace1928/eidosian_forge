from functools import partial
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from .retries import AWSRetry
from .tagging import boto3_tag_list_to_ansible_dict
def paginate_list_cloudfront_property(self, client_method, key, default_keyed, error, **kwargs):
    fail_if_error = kwargs.pop('fail_if_error', True)
    try:
        keyed = kwargs.pop('keyed', default_keyed)
        api_kwargs = snake_dict_to_camel_dict(kwargs, capitalize_first=True)
        result = _cloudfront_paginate_build_full_result(self.client, client_method, **api_kwargs)
        items = result.get(key, {}).get('Items', [])
        if keyed:
            items = cloudfront_facts_keyed_list_helper(items)
        return items
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        if not fail_if_error:
            raise
        self.module.fail_json_aws(e, msg=error)