from __future__ import (absolute_import, division, print_function)
import json
import re
import time
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params, \
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import wait_for_redfish_reboot_job, \
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def wait_for_redfish_idrac_reset(module, redfish_obj, wait_time_sec, interval=30):
    time.sleep(interval // 2)
    msg = RESET_UNTRACK
    wait = wait_time_sec
    track_failed = True
    resetting = False
    while wait > 0 and track_failed:
        try:
            redfish_obj.invoke_request('GET', MANAGERS_URI, api_timeout=120)
            msg = RESET_SUCCESS
            track_failed = False
            break
        except HTTPError as err:
            if err.getcode() == 401:
                new_redfish_obj = Redfish(module.params, req_session=True)
                sid, token = require_session(new_redfish_obj, module)
                redfish_obj.session_id = sid
                redfish_obj._headers.update({'X-Auth-Token': token})
                track_failed = False
                if not resetting:
                    resetting = True
                break
            time.sleep(interval)
            wait -= interval
            resetting = True
        except URLError:
            time.sleep(interval)
            wait -= interval
            if not resetting:
                resetting = True
        except Exception:
            time.sleep(interval)
            wait -= interval
            resetting = True
    return (track_failed, resetting, msg)