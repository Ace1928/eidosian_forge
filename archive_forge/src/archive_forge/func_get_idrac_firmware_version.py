from __future__ import (absolute_import, division, print_function)
import time
from datetime import datetime
import re
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def get_idrac_firmware_version(idrac):
    firm_version = idrac.invoke_request(method='GET', uri=GET_IDRAC_FIRMWARE_VER_URI)
    return firm_version.json_data.get('FirmwareVersion', '')