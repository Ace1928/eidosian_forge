from copy import deepcopy
from functools import wraps
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def set_wait(self, wait):
    if wait is None:
        return False
    if wait == self._wait:
        return False
    self._wait = wait
    return True