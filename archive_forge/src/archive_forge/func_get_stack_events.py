import json
import time
import traceback
import uuid
from hashlib import sha1
from ansible.module_utils._text import to_bytes
from ansible.module_utils._text import to_native
from ansible_collections.amazon.aws.plugins.module_utils.botocore import boto_exception
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
def get_stack_events(cfn, stack_name, events_limit, token_filter=None):
    """This event data was never correct, it worked as a side effect. So the v2.3 format is different."""
    ret = {'events': [], 'log': []}
    try:
        pg = cfn.get_paginator('describe_stack_events').paginate(StackName=stack_name, PaginationConfig={'MaxItems': events_limit})
        if token_filter is not None:
            events = list(retry_decorator(pg.search)(f"StackEvents[?ClientRequestToken == '{token_filter}']"))
        else:
            events = list(pg.search('StackEvents[*]'))
    except is_boto3_error_message('does not exist'):
        ret['log'].append('Stack does not exist.')
        return ret
    except (botocore.exceptions.ValidationError, botocore.exceptions.ClientError) as err:
        error_msg = boto_exception(err)
        ret['log'].append('Unknown error: ' + str(error_msg))
        return ret
    for e in events:
        eventline = f'StackEvent {e['ResourceType']} {e['LogicalResourceId']} {e['ResourceStatus']}'
        ret['events'].append(eventline)
        if e['ResourceStatus'].endswith('FAILED'):
            failure = f'{e['ResourceType']} {e['LogicalResourceId']} {e['ResourceStatus']}: {e['ResourceStatusReason']}'
            ret['log'].append(failure)
    return ret