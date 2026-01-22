from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_or_update_bucket_cors(connection, module):
    name = module.params.get('name')
    rules = module.params.get('rules', [])
    changed = False
    try:
        current_camel_rules = connection.get_bucket_cors(Bucket=name)['CORSRules']
    except ClientError:
        current_camel_rules = []
    new_camel_rules = snake_dict_to_camel_dict(rules, capitalize_first=True)
    if compare_policies(new_camel_rules, current_camel_rules):
        changed = True
    if changed:
        try:
            cors = connection.put_bucket_cors(Bucket=name, CORSConfiguration={'CORSRules': new_camel_rules})
        except (BotoCoreError, ClientError) as e:
            module.fail_json_aws(e, msg=f'Unable to update CORS for bucket {name}')
    module.exit_json(changed=changed, name=name, rules=rules)