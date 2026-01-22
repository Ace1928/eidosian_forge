from __future__ import unicode_literals
import sys
import os
import requests
import requests.auth
import warnings
from winrm.exceptions import InvalidCredentialsError, WinRMError, WinRMTransportError
from winrm.encryption import Encryption
def setup_encryption(self):
    request = requests.Request('POST', self.endpoint, data=None)
    prepared_request = self.session.prepare_request(request)
    self._send_message_request(prepared_request, '')
    self.encryption = Encryption(self.session, self.auth_method)