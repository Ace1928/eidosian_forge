from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def wait_ontap_image_upgrade_complete(self, rest_api, headers, desired):
    retry_count = 65
    if self.parameters['is_ha'] is True:
        retry_count *= 2
    for count in range(retry_count):
        we, err = self.get_working_environment_property(rest_api, headers, ['status', 'ontapClusterProperties'])
        if err is not None:
            return (False, 'Error: get_working_environment_property failed: %s' % str(err))
        if we['status']['status'] != 'UPDATING' and we['ontapClusterProperties']['ontapVersion'] != '':
            if we['ontapClusterProperties']['ontapVersion'] in desired:
                return (True, None)
        time.sleep(60)
    return (False, 'Error: Taking too long for CVO to be active or not properly setup')