import datetime
import functools
import time
from copy import deepcopy
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def wait_for_domain_status(client, module, domain_name, waiter_name):
    if not module.params['wait']:
        return
    timeout = module.params['wait_timeout']
    deadline = time.time() + timeout
    status_msg = ''
    while time.time() < deadline:
        status = get_domain_status(client, module, domain_name)
        if status is None:
            status_msg = 'Not Found'
            if waiter_name == 'domain_deleted':
                return
        else:
            status_msg = 'Created: {0}. Processing: {1}. UpgradeProcessing: {2}'.format(status['Created'], status['Processing'], status['UpgradeProcessing'])
            if waiter_name == 'domain_available' and status['Created'] and (not status['Processing']) and (not status['UpgradeProcessing']):
                return
        time.sleep(15)
    module.fail_json(msg=f"Timeout waiting for wait state '{waiter_name}'. {status_msg}")