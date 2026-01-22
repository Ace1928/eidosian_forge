from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.urls import ConnectionError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
import ansible_collections.community.general.plugins.module_utils.influxdb as influx
def drop_user(module, client, user_name):
    if not module.check_mode:
        try:
            client.drop_user(user_name)
        except influx.exceptions.InfluxDBClientError as e:
            module.fail_json(msg=e.content)
    module.exit_json(changed=True)