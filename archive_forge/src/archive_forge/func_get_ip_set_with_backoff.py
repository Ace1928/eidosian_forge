from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .retries import AWSRetry
from .waiters import get_waiter
@AWSRetry.jittered_backoff(delay=5)
def get_ip_set_with_backoff(client, ip_set_id):
    return client.get_ip_set(IPSetId=ip_set_id)['IPSet']