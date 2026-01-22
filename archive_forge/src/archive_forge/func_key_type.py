from __future__ import absolute_import, division, print_function
import os
from base64 import b64encode, b64decode
from getpass import getuser
from socket import gethostname
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
@property
def key_type(self):
    """Returns the key type of this key pair"""
    return self.__asym_keypair.key_type