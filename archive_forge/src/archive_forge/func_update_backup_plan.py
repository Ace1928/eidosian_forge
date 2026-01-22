import json
from datetime import datetime
from typing import Optional
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.backup import get_plan_details
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def update_backup_plan(module: AnsibleAWSModule, client, update_params: dict) -> dict:
    """
    Updates a backup plan.

    module : AnsibleAWSModule object
    client : boto3 backup client connection object
    update_params : The boto3 backup client parameters to update a backup plan
    """
    try:
        response = client.update_backup_plan(**update_params)
    except (BotoCoreError, ClientError) as err:
        module.fail_json_aws(err, msg='Failed to update backup plan {err}')
    return response