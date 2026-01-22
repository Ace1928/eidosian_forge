from __future__ import absolute_import, division, print_function
import traceback
from time import sleep
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def get_build_status(self):
    try:
        response = self.server.get_build_info(self.name, self.build_number)
        return response
    except jenkins.JenkinsException as e:
        response = {}
        response['result'] = 'ABSENT'
        return response
    except Exception as e:
        self.module.fail_json(msg='Unable to fetch build information, %s' % to_native(e), exception=traceback.format_exc())