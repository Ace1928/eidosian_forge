import json
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_or_update_identity_policy(connection, module):
    identity = module.params.get('identity')
    policy_name = module.params.get('policy_name')
    required_policy = module.params.get('policy')
    required_policy_dict = json.loads(required_policy)
    changed = False
    policy = get_identity_policy(connection, module, identity, policy_name)
    policy_dict = json.loads(policy) if policy else None
    if compare_policies(policy_dict, required_policy_dict):
        changed = True
        try:
            if not module.check_mode:
                connection.put_identity_policy(Identity=identity, PolicyName=policy_name, Policy=required_policy, aws_retry=True)
        except (BotoCoreError, ClientError) as e:
            module.fail_json_aws(e, msg=f'Failed to put identity policy {policy_name}')
    try:
        policies_present = connection.list_identity_policies(Identity=identity, aws_retry=True)['PolicyNames']
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e, msg='Failed to list identity policies')
    if policy_name is not None and policy_name not in policies_present:
        policies_present = list(policies_present)
        policies_present.append(policy_name)
    module.exit_json(changed=changed, policies=policies_present)