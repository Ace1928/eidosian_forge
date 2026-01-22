from __future__ import absolute_import, division, print_function
import hashlib
import re
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.six.moves import zip
def get_block_config(self, path):
    block = self.get_block(path)
    return dumps(block, 'block')