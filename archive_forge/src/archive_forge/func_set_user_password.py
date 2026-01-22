from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.urls import ConnectionError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
import ansible_collections.community.general.plugins.module_utils.influxdb as influx
def set_user_password(module, client, user_name, user_password):
    if not module.check_mode:
        try:
            client.set_user_password(user_name, user_password)
        except ConnectionError as e:
            module.fail_json(msg=to_native(e))