from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .retries import AWSRetry
from .waiters import get_waiter
@AWSRetry.jittered_backoff(delay=5)
def list_web_acls_with_backoff(client):
    paginator = client.get_paginator('list_web_acls')
    return paginator.paginate().build_full_result()['WebACLs']