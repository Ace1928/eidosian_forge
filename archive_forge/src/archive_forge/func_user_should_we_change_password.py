from __future__ import absolute_import, division, print_function
import hmac
import itertools
import re
import traceback
from base64 import b64decode
from hashlib import md5, sha256
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils import \
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def user_should_we_change_password(current_role_attrs, user, password, encrypted):
    """Check if we should change the user's password.

    Compare the proposed password with the existing one, comparing
    hashes if encrypted. If we can't access it assume yes.
    """
    if current_role_attrs is None:
        return True
    pwchanging = False
    if password is not None:
        current_password = current_role_attrs['rolpassword']
        if isinstance(current_password, bytes):
            current_password = current_password.decode('ascii')
        if password == '':
            if current_password is not None:
                pwchanging = True
        elif re.match(SCRAM_SHA256_REGEX, password):
            if password != current_password:
                pwchanging = True
        elif current_password is not None and pbkdf2_found and re.match(SCRAM_SHA256_REGEX, current_password):
            r = re.match(SCRAM_SHA256_REGEX, current_password)
            try:
                it = int(r.group(1))
                salt = b64decode(r.group(2))
                server_key = b64decode(r.group(4))
                normalized_password = saslprep.saslprep(to_text(password))
                salted_password = pbkdf2_hmac('sha256', to_bytes(normalized_password), salt, it)
                server_key_verifier = hmac.new(salted_password, digestmod=sha256)
                server_key_verifier.update(b'Server Key')
                if server_key_verifier.digest() != server_key:
                    pwchanging = True
            except Exception:
                pwchanging = True
        elif password.startswith('md5') and len(password) == 32 + 3 or encrypted == 'UNENCRYPTED':
            if password != current_password:
                pwchanging = True
        elif encrypted == 'ENCRYPTED':
            hashed_password = 'md5{0}'.format(md5(to_bytes(password) + to_bytes(user)).hexdigest())
            if hashed_password != current_password:
                pwchanging = True
    return pwchanging