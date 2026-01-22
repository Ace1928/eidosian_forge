import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_identity_notifications(connection, module, identity, retries=0, retryDelay=10):
    for attempt in range(0, retries + 1):
        try:
            response = connection.get_identity_notification_attributes(Identities=[identity], aws_retry=True)
        except (BotoCoreError, ClientError) as e:
            module.fail_json_aws(e, msg=f'Failed to retrieve identity notification attributes for {identity}')
        notification_attributes = response['NotificationAttributes']
        if identity in notification_attributes:
            break
        elif len(notification_attributes) != 0:
            module.fail_json(msg='Unexpected identity found in notification attributes, expected {0} but got {1!r}.'.format(identity, notification_attributes.keys()))
        time.sleep(retryDelay)
    if identity not in notification_attributes:
        return None
    return notification_attributes[identity]