from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def upgrade_ontap_image(self, rest_api, headers, desired):
    dummy, err = self.set_config_flag(rest_api, headers)
    if err is not None:
        return (False, err)
    dummy, err = self.do_ontap_image_upgrade(rest_api, headers, desired)
    if err is not None:
        return (False, err)
    dummy, err = self.wait_ontap_image_upgrade_complete(rest_api, headers, desired)
    return (err is None, err)