import time
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.route53 import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.route53 import manage_tags
def find_zones(zone_in, private_zone):
    try:
        results = _list_zones()
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e, msg='Could not list current hosted zones')
    zones = []
    for r53zone in results['HostedZones']:
        if r53zone['Name'] != zone_in:
            continue
        if r53zone['Config']['PrivateZone'] and private_zone or (not r53zone['Config']['PrivateZone'] and (not private_zone)):
            zones.append(r53zone)
    return zones