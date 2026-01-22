from __future__ import absolute_import, division, print_function
import time
from .common import (
from .constants import (
from .icontrol import iControlRestSession
def connect_via_basic_auth(self):
    url = 'https://{0}:{1}/mgmt/tm/sys'.format(self.provider['server'], self.provider['server_port'])
    session = iControlRestSession(url_username=self.provider['user'], url_password=self.provider['password'], validate_certs=self.provider['validate_certs'])
    response = session.get(url, headers=self.headers)
    if response.status not in [200]:
        if b'Configuration Utility restarting...' in response.content and self.retries < 3:
            time.sleep(30)
            self.retries += 1
            return self.connect_via_basic_auth()
        else:
            self.retries = 0
            return (None, response.content)
    self.retries = 0
    return (session, None)