from __future__ import absolute_import, division, print_function
import os
import tempfile
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import string_types
from ansible.module_utils._text import to_bytes
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def encrypt_password_change_file(self, public_key, password):
    pub = serialization.load_pem_public_key(to_bytes(public_key), backend=default_backend())
    message = to_bytes('{0}\n{0}\n'.format(password))
    ciphertext = pub.encrypt(message, padding.PKCS1v15())
    return BytesIO(ciphertext)