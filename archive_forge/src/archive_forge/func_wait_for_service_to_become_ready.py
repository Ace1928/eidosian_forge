from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
def wait_for_service_to_become_ready(module, auth, service_id, wait_timeout):
    import time
    start_time = time.time()
    while time.time() - start_time < wait_timeout:
        try:
            status_result = open_url(auth.url + '/service/' + str(service_id), method='GET', force_basic_auth=True, url_username=auth.user, url_password=auth.password)
        except Exception as e:
            module.fail_json(msg='Request for service status has failed. Error message: ' + str(e))
        status_result = module.from_json(status_result.read())
        service_state = status_result['DOCUMENT']['TEMPLATE']['BODY']['state']
        if service_state in [STATES.index('RUNNING'), STATES.index('COOLDOWN')]:
            return status_result['DOCUMENT']
        elif service_state not in [STATES.index('PENDING'), STATES.index('DEPLOYING'), STATES.index('SCALING')]:
            log_message = ''
            for log_info in status_result['DOCUMENT']['TEMPLATE']['BODY']['log']:
                if log_info['severity'] == 'E':
                    log_message = log_message + log_info['message']
                    break
            module.fail_json(msg='Deploying is unsuccessful. Service state: ' + STATES[service_state] + '. Error message: ' + log_message)
        time.sleep(1)
    module.fail_json(msg='Wait timeout has expired')