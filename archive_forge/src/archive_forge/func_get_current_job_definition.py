from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.batch import cc
from ansible_collections.amazon.aws.plugins.module_utils.batch import set_api_params
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_current_job_definition(module, batch_client):
    try:
        environments = batch_client.describe_job_definitions(jobDefinitionName=module.params['job_definition_name'])
        if len(environments['jobDefinitions']) > 0:
            latest_revision = max(map(lambda d: d['revision'], environments['jobDefinitions']))
            latest_definition = next((x for x in environments['jobDefinitions'] if x['revision'] == latest_revision), None)
            return latest_definition
        return None
    except ClientError:
        return None