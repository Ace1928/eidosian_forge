import ctypes
import json
import logging
import os
import sys
import cffi  # type: ignore
import six
from google.auth import exceptions
def set_up_custom_key(self):
    self._cert = get_cert(self._signer_lib, self._enterprise_cert_file_path)
    self._sign_callback = get_sign_callback(self._signer_lib, self._enterprise_cert_file_path)