import datetime
import time
from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.botocore import normalize_boto3_result
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def merge_transitions(updated_rule, updating_rule):
    updated_transitions = {}
    updating_transitions = {}
    for transition in updated_rule.get('Transitions', []):
        updated_transitions[transition['StorageClass']] = transition
    for transition in updating_rule.get('Transitions', []):
        updating_transitions[transition['StorageClass']] = transition
    for storage_class, transition in updating_transitions.items():
        if updated_transitions.get(storage_class) is None:
            updated_rule['Transitions'].append(transition)