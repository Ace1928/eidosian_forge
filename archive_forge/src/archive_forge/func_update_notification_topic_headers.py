import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def update_notification_topic_headers(connection, module, identity, identity_notifications, notification_type):
    arg_dict = module.params.get(notification_type.lower() + '_notifications')
    header_key = 'HeadersIn' + notification_type + 'NotificationsEnabled'
    if identity_notifications is None:
        current = False
    elif header_key in identity_notifications:
        current = identity_notifications[header_key]
    else:
        current = False
    if arg_dict is not None and 'include_headers' in arg_dict:
        required = arg_dict['include_headers']
    else:
        required = False
    if current != required:
        try:
            if not module.check_mode:
                connection.set_identity_headers_in_notifications_enabled(Identity=identity, NotificationType=notification_type, Enabled=required, aws_retry=True)
        except (BotoCoreError, ClientError) as e:
            module.fail_json_aws(e, msg=f'Failed to set identity headers in notification for {identity} {notification_type}')
        return True
    return False