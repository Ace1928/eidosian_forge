from __future__ import absolute_import, division, print_function
import json
import re
import sys
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils._text import to_native
def get_nitro_client(module):
    from nssrc.com.citrix.netscaler.nitro.service.nitro_service import nitro_service
    client = nitro_service(module.params['nsip'], module.params['nitro_protocol'])
    client.set_credential(module.params['nitro_user'], module.params['nitro_pass'])
    client.timeout = float(module.params['nitro_timeout'])
    client.certvalidation = module.params['validate_certs']
    return client