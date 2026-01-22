from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def stop_execution(module, sfn_client):
    cause = module.params.get('cause')
    error = module.params.get('error')
    execution_arn = module.params.get('execution_arn')
    try:
        execution_status = sfn_client.describe_execution(executionArn=execution_arn)['status']
        if execution_status != 'RUNNING':
            check_mode(module, msg='State machine execution is not running.', changed=False)
            module.exit_json(changed=False)
        check_mode(module, msg='State machine execution would be stopped.', changed=True)
        res = sfn_client.stop_execution(executionArn=execution_arn, cause=cause, error=error)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to stop execution.')
    module.exit_json(changed=True, **camel_dict_to_snake_dict(res))