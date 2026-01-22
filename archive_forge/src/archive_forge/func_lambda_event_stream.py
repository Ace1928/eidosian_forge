import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import boto3_conn
from ansible_collections.amazon.aws.plugins.module_utils.botocore import get_aws_connection_info
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def lambda_event_stream(module, aws):
    """
    Adds, updates or deletes lambda stream (DynamoDb, Kinesis) event notifications.
    :param module:
    :param aws:
    :return:
    """
    client = aws.client('lambda')
    facts = dict()
    changed = False
    current_state = 'absent'
    state = module.params['state']
    api_params = dict(FunctionName=module.params['lambda_function_arn'])
    source_params = module.params['source_params']
    source_arn = source_params.get('source_arn')
    if source_arn:
        api_params.update(EventSourceArn=source_arn)
    else:
        module.fail_json(msg="Source parameter 'source_arn' is required for stream event notification.")
    batch_size = source_params.get('batch_size')
    if batch_size:
        try:
            source_params['batch_size'] = int(batch_size)
        except ValueError:
            module.fail_json(msg=f"Source parameter 'batch_size' must be an integer, found: {source_params['batch_size']}")
    source_param_enabled = module.boolean(source_params.get('enabled', 'True'))
    try:
        facts = client.list_event_source_mappings(**api_params)['EventSourceMappings']
        if facts:
            current_state = 'present'
    except ClientError as e:
        module.fail_json(msg=f'Error retrieving stream event notification configuration: {e}')
    if state == 'present':
        if current_state == 'absent':
            starting_position = source_params.get('starting_position')
            if starting_position:
                api_params.update(StartingPosition=starting_position)
            elif module.params.get('event_source') == 'sqs':
                pass
            else:
                module.fail_json(msg="Source parameter 'starting_position' is required for stream event notification.")
            if source_arn:
                api_params.update(Enabled=source_param_enabled)
            if source_params.get('batch_size'):
                api_params.update(BatchSize=source_params.get('batch_size'))
            if source_params.get('function_response_types'):
                api_params.update(FunctionResponseTypes=source_params.get('function_response_types'))
            try:
                if not module.check_mode:
                    facts = client.create_event_source_mapping(**api_params)
                changed = True
            except (ClientError, ParamValidationError, MissingParametersError) as e:
                module.fail_json(msg=f'Error creating stream source event mapping: {e}')
        else:
            api_params = dict(FunctionName=module.params['lambda_function_arn'])
            current_mapping = facts[0]
            api_params.update(UUID=current_mapping['UUID'])
            mapping_changed = False
            if source_params.get('batch_size') and source_params['batch_size'] != current_mapping['BatchSize']:
                api_params.update(BatchSize=source_params['batch_size'])
                mapping_changed = True
            if source_param_enabled is not None:
                if source_param_enabled:
                    if current_mapping['State'] not in ('Enabled', 'Enabling'):
                        api_params.update(Enabled=True)
                        mapping_changed = True
                elif current_mapping['State'] not in ('Disabled', 'Disabling'):
                    api_params.update(Enabled=False)
                    mapping_changed = True
            if mapping_changed:
                try:
                    if not module.check_mode:
                        facts = client.update_event_source_mapping(**api_params)
                    changed = True
                except (ClientError, ParamValidationError, MissingParametersError) as e:
                    module.fail_json(msg=f'Error updating stream source event mapping: {e}')
    elif current_state == 'present':
        api_params = dict(UUID=facts[0]['UUID'])
        try:
            if not module.check_mode:
                facts = client.delete_event_source_mapping(**api_params)
            changed = True
        except (ClientError, ParamValidationError, MissingParametersError) as e:
            module.fail_json(msg=f'Error removing stream source event mapping: {e}')
    return camel_dict_to_snake_dict(dict(changed=changed, events=facts))