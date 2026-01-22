from __future__ import absolute_import, division, print_function
import abc
import binascii
import os
from base64 import b64encode
from datetime import datetime
from hashlib import sha256
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_text
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import convert_relative_to_datetime
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
def within_range(self, valid_at):
    if valid_at is not None:
        valid_at_datetime = self.to_datetime(valid_at)
        return self._valid_from <= valid_at_datetime <= self._valid_to
    return True