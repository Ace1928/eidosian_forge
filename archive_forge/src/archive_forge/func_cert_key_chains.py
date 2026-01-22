from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import string_types
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
@property
def cert_key_chains(self):
    result = []
    if self.client_ssl_profile is None:
        return None
    if 'cert_key_chain' not in self.client_ssl_profile:
        return None
    kc = self.client_ssl_profile['cert_key_chain']
    if isinstance(kc, string_types) and kc != 'inherit':
        raise F5ModuleError("Only the 'inherit' setting is available when 'cert_key_chain' is a string.")
    if not isinstance(kc, list):
        raise F5ModuleError("The value of 'cert_key_chain' is not one of the supported types.")
    cert_references = self._get_cert_references()
    key_references = self._get_key_references()
    for idx, x in enumerate(kc):
        tmp = dict(name='clientssl{0}'.format(idx))
        if 'cert' not in x:
            raise F5ModuleError("A 'cert' option is required when specifying the 'cert_key_chain' parameter..")
        elif x['cert'] not in cert_references:
            raise F5ModuleError("The specified 'cert' was not found. Did you specify its full path?")
        else:
            key = x['cert']
            tmp['certReference'] = dict(link=cert_references[key], fullPath=key)
        if 'key' not in x:
            raise F5ModuleError("A 'key' option is required when specifying the 'cert_key_chain' parameter..")
        elif x['key'] not in key_references:
            raise F5ModuleError("The specified 'key' was not found. Did you specify its full path?")
        else:
            key = x['key']
            tmp['keyReference'] = dict(link=key_references[key], fullPath=key)
        if 'chain' in x and x['chain'] not in cert_references:
            raise F5ModuleError("The specified 'key' was not found. Did you specify its full path?")
        else:
            key = x['chain']
            tmp['chainReference'] = dict(link=cert_references[key], fullPath=key)
        if 'passphrase' in x:
            tmp['passphrase'] = x['passphrase']
        result.append(tmp)
    return result