from __future__ import absolute_import, division, print_function
import base64
import binascii
import re
from ansible.module_utils.basic import AnsibleModule, AVAILABLE_HASH_ALGORITHMS
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.community.hrobot.plugins.module_utils.robot import (
def normalize_fingerprint(fingerprint, size=16):
    if ':' in fingerprint:
        fingerprint = fingerprint.split(':')
    else:
        fingerprint = [fingerprint[i:i + 2] for i in range(0, len(fingerprint), 2)]
    if len(fingerprint) != size:
        raise FingerprintError('Fingerprint must consist of {0} 8-bit hex numbers: got {1} 8-bit hex numbers instead'.format(size, len(fingerprint)))
    for i, part in enumerate(fingerprint):
        new_part = part.lower()
        if len(new_part) < 2:
            new_part = '0{0}'.format(new_part)
        if not FINGERPRINT_PART.match(new_part):
            raise FingerprintError('Fingerprint must consist of {0} 8-bit hex numbers: number {1} is invalid: "{2}"'.format(size, i + 1, part))
        fingerprint[i] = new_part
    return ':'.join(fingerprint)