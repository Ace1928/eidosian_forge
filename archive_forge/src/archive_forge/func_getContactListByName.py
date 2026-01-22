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
def getContactListByName(self, name):
    if not self.contactList_map:
        self._instMap('contactList')
    return self.getContactList(self.contactList_map.get(name, 0))