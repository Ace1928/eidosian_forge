import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
from google.auth import transport
def test_apply_impersonation_without_quota_project_id(self):
    expire_time = (_helpers.utcnow().replace(microsecond=0) + datetime.timedelta(seconds=3600)).isoformat('T') + 'Z'
    impersonation_response = {'accessToken': 'SA_ACCESS_TOKEN', 'expireTime': expire_time}
    request = self.make_mock_request(status=http_client.OK, data=self.SUCCESS_RESPONSE.copy(), impersonation_status=http_client.OK, impersonation_data=impersonation_response)
    credentials = self.make_credentials(service_account_impersonation_url=self.SERVICE_ACCOUNT_IMPERSONATION_URL, scopes=self.SCOPES)
    headers = {}
    credentials.refresh(request)
    credentials.apply(headers)
    assert headers == {'authorization': 'Bearer {}'.format(impersonation_response['accessToken'])}