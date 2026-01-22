import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_or_update_identity(connection, module, region, account_id):
    identity = module.params.get('identity')
    changed = False
    verification_attributes = get_verification_attributes(connection, module, identity)
    if verification_attributes is None:
        try:
            if not module.check_mode:
                if '@' in identity:
                    connection.verify_email_identity(EmailAddress=identity, aws_retry=True)
                else:
                    connection.verify_domain_identity(Domain=identity, aws_retry=True)
        except (BotoCoreError, ClientError) as e:
            module.fail_json_aws(e, msg=f'Failed to verify identity {identity}')
        if module.check_mode:
            verification_attributes = {'VerificationStatus': 'Pending'}
        else:
            verification_attributes = get_verification_attributes(connection, module, identity, retries=4)
        changed = True
    elif verification_attributes['VerificationStatus'] not in ('Pending', 'Success'):
        module.fail_json(msg='Identity ' + identity + ' in bad status ' + verification_attributes['VerificationStatus'], verification_attributes=camel_dict_to_snake_dict(verification_attributes))
    if verification_attributes is None:
        module.fail_json(msg='Unable to load identity verification attributes after registering identity.')
    notifications_changed, notification_attributes = update_identity_notifications(connection, module)
    changed |= notifications_changed
    if notification_attributes is None:
        module.fail_json(msg='Unable to load identity notification attributes.')
    identity_arn = 'arn:aws:ses:' + region + ':' + account_id + ':identity/' + identity
    module.exit_json(changed=changed, identity=identity, identity_arn=identity_arn, verification_attributes=camel_dict_to_snake_dict(verification_attributes), notification_attributes=camel_dict_to_snake_dict(notification_attributes))