from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
import re
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
def get_sgt_by_id(self, id):
    if not id:
        return None
    try:
        result = self.ise.exec(family='sgt', function='get_security_group_by_id', params={'id': id}, handle_func_exception=False).response['Sgt']
    except (TypeError, AttributeError) as e:
        self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
    except Exception:
        result = None
    return result