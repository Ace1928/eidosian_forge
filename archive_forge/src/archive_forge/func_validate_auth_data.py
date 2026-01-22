from __future__ import absolute_import, division, print_function
from . import errors, http
def validate_auth_data(self, username, password):
    resp = http.request('GET', '{0}/auth/test'.format(self.address), force_basic_auth=True, url_username=username, url_password=password, validate_certs=self.verify, ca_path=self.ca_path)
    if resp.status not in (200, 401):
        raise errors.SensuError('Authentication test returned status {0}'.format(resp.status))
    return resp.status == 200