import time
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.route53 import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.route53 import manage_tags
def delete_hosted_id(hosted_zone_id, matching_zones):
    if hosted_zone_id == 'all':
        deleted = []
        for z in matching_zones:
            deleted.append(z['Id'])
            if not module.check_mode:
                try:
                    client.delete_hosted_zone(Id=z['Id'])
                except (BotoCoreError, ClientError) as e:
                    module.fail_json_aws(e, msg=f'Could not delete hosted zone {z['Id']}')
        changed = True
        msg = f'Successfully deleted zones: {deleted}'
    elif hosted_zone_id in [zo['Id'].replace('/hostedzone/', '') for zo in matching_zones]:
        if not module.check_mode:
            try:
                client.delete_hosted_zone(Id=hosted_zone_id)
            except (BotoCoreError, ClientError) as e:
                module.fail_json_aws(e, msg=f'Could not delete hosted zone {hosted_zone_id}')
        changed = True
        msg = f'Successfully deleted zone: {hosted_zone_id}'
    else:
        changed = False
        msg = f'There is no zone to delete that matches hosted_zone_id {hosted_zone_id}.'
    return (changed, msg)