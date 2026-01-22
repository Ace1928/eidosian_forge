from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def check_app(ebs, app, module):
    app_name = module.params['app_name']
    description = module.params['description']
    state = module.params['state']
    terminate_by_force = module.params['terminate_by_force']
    result = {}
    if state == 'present' and app is None:
        result = dict(changed=True, output='App would be created')
    elif state == 'present' and app.get('Description', None) != description:
        result = dict(changed=True, output='App would be updated', app=app)
    elif state == 'present' and app.get('Description', None) == description:
        result = dict(changed=False, output='App is up-to-date', app=app)
    elif state == 'absent' and app is None:
        result = dict(changed=False, output='App does not exist', app={})
    elif state == 'absent' and app is not None:
        result = dict(changed=True, output='App will be deleted', app=app)
    elif state == 'absent' and app is not None and (terminate_by_force is True):
        result = dict(changed=True, output='Running environments terminated before the App will be deleted', app=app)
    module.exit_json(**result)