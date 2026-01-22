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
def create_changeset(module, stack_params, cfn, events_limit):
    if 'TemplateBody' not in stack_params and 'TemplateURL' not in stack_params:
        module.fail_json(msg="Either 'template' or 'template_url' is required.")
    if module.params['changeset_name'] is not None:
        stack_params['ChangeSetName'] = module.params['changeset_name']
    stack_params.pop('ClientRequestToken', None)
    try:
        changeset_name = build_changeset_name(stack_params)
        stack_params['ChangeSetName'] = changeset_name
        pending_changesets = list_changesets(cfn, stack_params['StackName'])
        if changeset_name in pending_changesets:
            warning = f'WARNING: {len(pending_changesets)} pending changeset(s) exist(s) for this stack!'
            result = dict(changed=False, output=f'ChangeSet {changeset_name} already exists.', warnings=[warning])
        else:
            cs = cfn.create_change_set(aws_retry=True, **stack_params)
            time_end = time.time() + 600
            while time.time() < time_end:
                try:
                    newcs = cfn.describe_change_set(aws_retry=True, ChangeSetName=cs['Id'])
                except botocore.exceptions.BotoCoreError as err:
                    module.fail_json_aws(err)
                if newcs['Status'] == 'CREATE_PENDING' or newcs['Status'] == 'CREATE_IN_PROGRESS':
                    time.sleep(1)
                elif newcs['Status'] == 'FAILED' and ("The submitted information didn't contain changes" in newcs['StatusReason'] or 'No updates are to be performed' in newcs['StatusReason']):
                    cfn.delete_change_set(aws_retry=True, ChangeSetName=cs['Id'])
                    result = dict(changed=False, output='The created Change Set did not contain any changes to this stack and was deleted.')
                    return result
                else:
                    break
                time.sleep(1)
            result = stack_operation(module, cfn, stack_params['StackName'], 'CREATE_CHANGESET', events_limit)
            result['change_set_id'] = cs['Id']
            result['warnings'] = [f'Created changeset named {changeset_name} for stack {stack_params['StackName']}', f'You can execute it using: aws cloudformation execute-change-set --change-set-name {cs['Id']}', 'NOTE that dependencies on this stack might fail due to pending changes!']
    except is_boto3_error_message('No updates are to be performed.'):
        result = dict(changed=False, output='Stack is already up-to-date.')
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as err:
        module.fail_json_aws(err, msg='Failed to create change set')
    if not result:
        module.fail_json(msg='empty result')
    return result