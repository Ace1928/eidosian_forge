from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.urls import ConnectionError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
import ansible_collections.community.general.plugins.module_utils.influxdb as influx
def check_user_password(module, client, user_name, user_password):
    try:
        client.switch_user(user_name, user_password)
        client.get_list_users()
    except influx.exceptions.InfluxDBClientError as e:
        if e.code == 401:
            return False
    except ConnectionError as e:
        module.fail_json(msg=to_native(e))
    finally:
        client.switch_user(module.params['username'], module.params['password'])
    return True