from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .retries import AWSRetry
from .waiters import get_waiter
@AWSRetry.jittered_backoff(delay=5)
def get_web_acl_with_backoff(client, web_acl_id):
    return client.get_web_acl(WebACLId=web_acl_id)['WebACL']