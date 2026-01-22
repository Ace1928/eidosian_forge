import datetime
import json
import httplib2
from google.auth import aws
from google.auth import external_account
from google.auth import external_account_authorized_user
from google.auth import identity_pool
from google.auth import pluggable
from gslib.tests import testcase
from gslib.utils.wrapped_credentials import WrappedCredentials
import oauth2client
from six import add_move, MovedModule
from six.moves import mock
@mock.patch.object(httplib2, 'Http', autospec=True)
def testWrappedCredentialUsageExternalAccountAuthorizedUser(self, http):
    http.return_value.request.return_value = (RESPONSE, CONTENT)
    req = http.return_value.request
    creds = WrappedCredentials(external_account_authorized_user.Credentials(audience='//iam.googleapis.com/locations/global/workforcePools/$WORKFORCE_POOL_ID/providers/$PROVIDER_ID', refresh_token='refreshToken', token_url='https://sts.googleapis.com/v1/oauth/token', token_info_url='https://sts.googleapis.com/v1/instrospect', client_id='clientId', client_secret='clientSecret'))

    def _refresh_token_side_effect(*args, **kwargs):
        del args, kwargs
        creds._base.token = ACCESS_TOKEN
    creds._base.refresh = mock.Mock(side_effect=_refresh_token_side_effect)
    http = oauth2client.transport.get_http_object()
    creds.authorize(http)
    _, content = http.request(uri='google.com')
    self.assertEqual(content, CONTENT)
    creds._base.refresh.assert_called_once_with(mock.ANY)
    req.assert_called_once_with('google.com', method='GET', headers=HeadersWithAuth(ACCESS_TOKEN), body=None, connection_type=mock.ANY, redirections=mock.ANY)