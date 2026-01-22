from __future__ import absolute_import, division, print_function
import json
import hashlib
import hmac
import locale
from time import strftime, gmtime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.six import string_types
def updateMonitor(self, record_id, data):
    return self.query(self.monitor_url + '/' + str(record_id), 'PUT', data)