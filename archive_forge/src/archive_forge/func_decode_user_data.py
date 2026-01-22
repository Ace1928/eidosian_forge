import json
import re
import socket
import time
import zlib
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.urls import fetch_url
def decode_user_data(self, data):
    is_compressed = False
    if data.startswith(b'x\x9c') or data.startswith(b'\x1f\x8b'):
        is_compressed = True
    if is_compressed:
        try:
            decompressed = zlib.decompress(data, zlib.MAX_WBITS | 32)
            return self._decode(decompressed)
        except zlib.error:
            self.module.warn('Unable to decompress user-data using zlib, attempt to decode original user-data as UTF-8')
            return self._decode(data)
    else:
        return self._decode(data)