from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.influxdb import InfluxDb
from ansible.module_utils.common.text.converters import to_native
def find_retention_policy(module, client):
    database_name = module.params['database_name']
    policy_name = module.params['policy_name']
    hostname = module.params['hostname']
    retention_policy = None
    try:
        retention_policies = client.get_list_retention_policies(database=database_name)
        for policy in retention_policies:
            if policy['name'] == policy_name:
                retention_policy = policy
                break
    except requests.exceptions.ConnectionError as e:
        module.fail_json(msg='Cannot connect to database %s on %s : %s' % (database_name, hostname, to_native(e)))
    if retention_policy is not None:
        retention_policy['duration'] = parse_duration_literal(retention_policy['duration'], extended=True)
        retention_policy['shardGroupDuration'] = parse_duration_literal(retention_policy['shardGroupDuration'], extended=True)
    return retention_policy