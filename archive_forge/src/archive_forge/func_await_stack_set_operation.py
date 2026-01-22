import datetime
import itertools
import time
import uuid
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def await_stack_set_operation(module, cfn, stack_set_name, operation_id, max_wait):
    wait_start = datetime.datetime.now()
    operation = None
    for i in range(max_wait // 15):
        try:
            operation = cfn.describe_stack_set_operation(StackSetName=stack_set_name, OperationId=operation_id)
            if operation['StackSetOperation']['Status'] not in ('RUNNING', 'STOPPING'):
                break
        except is_boto3_error_code('StackSetNotFound'):
            pass
        except is_boto3_error_code('OperationNotFound'):
            pass
        time.sleep(15)
    if operation and operation['StackSetOperation']['Status'] not in ('FAILED', 'STOPPED'):
        await_stack_instance_completion(module, cfn, stack_set_name=stack_set_name, max_wait=int(max_wait - (datetime.datetime.now() - wait_start).total_seconds()))
    elif operation and operation['StackSetOperation']['Status'] in ('FAILED', 'STOPPED'):
        pass
    else:
        module.warn(f'Timed out waiting for operation {operation_id} on stack set {stack_set_name} after {max_wait} seconds. Returning unfinished operation')