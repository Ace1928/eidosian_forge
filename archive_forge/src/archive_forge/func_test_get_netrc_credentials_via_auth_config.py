from io import BytesIO
from .... import config, errors, osutils, tests
from .... import transport as _mod_transport
from ... import netrc_credential_store
def test_get_netrc_credentials_via_auth_config(self):
    ac_content = b'\n[host1]\nhost = host\nuser = joe\npassword_encoding = netrc\n'
    conf = config.AuthenticationConfig(_file=BytesIO(ac_content))
    credentials = conf.get_credentials('scheme', 'host', user='joe')
    self.assertIsNot(None, credentials)
    self.assertEqual('secret', credentials.get('password', None))