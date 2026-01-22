import json
from datetime import datetime
from typing import Optional
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.backup import get_plan_details
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def format_client_params(module: AnsibleAWSModule, plan: dict, tags: Optional[dict]=None, backup_plan_id: Optional[str]=None, operation: Optional[str]=None) -> dict:
    """
    Formats plan details to match boto3 backup client param expectations.

    module : AnsibleAWSModule object
    plan: Dict of plan details including name, rules, and advanced settings
    tags: Dict of plan tags
    backup_plan_id: ID of backup plan to update, only needed for update operation
    operation: Operation to add specific params for, either create or update
    """
    params = {'BackupPlan': snake_dict_to_camel_dict({k: v for k, v in plan.items() if v != 'backup_plan_name'}, capitalize_first=True)}
    if operation == 'create':
        if tags:
            params['BackupPlanTags'] = tags
        creator_request_id = module.params['creator_request_id']
        if creator_request_id:
            params['CreatorRequestId'] = creator_request_id
    elif operation == 'update':
        params['BackupPlanId'] = backup_plan_id
    return params