from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def manage_state_machine(state, sfn_client, module):
    state_machine_arn = get_state_machine_arn(sfn_client, module)
    if state == 'present':
        if state_machine_arn is None:
            create(sfn_client, module)
        else:
            update(state_machine_arn, sfn_client, module)
    elif state == 'absent':
        if state_machine_arn is not None:
            remove(state_machine_arn, sfn_client, module)
    check_mode(module, msg='State is up-to-date.')
    module.exit_json(changed=False, state_machine_arn=state_machine_arn)