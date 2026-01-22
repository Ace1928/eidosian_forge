import json
import re
from ansible.module_utils._text import to_native
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def policy_equal(module, current_statement):
    for param in ('action', 'principal', 'source_arn', 'source_account', 'event_source_token'):
        if module.params.get(param) != current_statement.get(param):
            return False
    return True