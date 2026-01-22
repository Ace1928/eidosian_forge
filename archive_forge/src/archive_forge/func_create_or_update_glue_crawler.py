from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_info
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_or_update_glue_crawler(connection, module, glue_crawler):
    """
    Create or update an AWS Glue crawler
    """
    changed = False
    params = dict()
    params['Name'] = module.params.get('name')
    params['Role'] = module.params.get('role')
    params['Targets'] = module.params.get('targets')
    if module.params.get('database_name') is not None:
        params['DatabaseName'] = module.params.get('database_name')
    if module.params.get('description') is not None:
        params['Description'] = module.params.get('description')
    if module.params.get('recrawl_policy') is not None:
        params['RecrawlPolicy'] = snake_dict_to_camel_dict(module.params.get('recrawl_policy'), capitalize_first=True)
    if module.params.get('role') is not None:
        params['Role'] = module.params.get('role')
    if module.params.get('schema_change_policy') is not None:
        params['SchemaChangePolicy'] = snake_dict_to_camel_dict(module.params.get('schema_change_policy'), capitalize_first=True)
    if module.params.get('table_prefix') is not None:
        params['TablePrefix'] = module.params.get('table_prefix')
    if module.params.get('targets') is not None:
        params['Targets'] = module.params.get('targets')
    if glue_crawler:
        if _compare_glue_crawler_params(params, glue_crawler):
            try:
                if not module.check_mode:
                    connection.update_crawler(aws_retry=True, **params)
                changed = True
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e)
    else:
        try:
            if not module.check_mode:
                connection.create_crawler(aws_retry=True, **params)
            changed = True
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e)
    glue_crawler = _get_glue_crawler(connection, module, params['Name'])
    changed |= ensure_tags(connection, module, glue_crawler)
    module.exit_json(changed=changed, **camel_dict_to_snake_dict(glue_crawler or {}, ignore_list=['SchemaChangePolicy', 'RecrawlPolicy', 'Targets']))