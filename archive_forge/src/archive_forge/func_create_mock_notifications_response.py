import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_mock_notifications_response(module):
    resp = {'ForwardingEnabled': module.params.get('feedback_forwarding')}
    for notification_type in ('Bounce', 'Complaint', 'Delivery'):
        arg_dict = module.params.get(notification_type.lower() + '_notifications')
        if arg_dict is not None and 'topic' in arg_dict:
            resp[notification_type + 'Topic'] = arg_dict['topic']
        header_key = 'HeadersIn' + notification_type + 'NotificationsEnabled'
        if arg_dict is not None and 'include_headers' in arg_dict:
            resp[header_key] = arg_dict['include_headers']
        else:
            resp[header_key] = False
    return resp