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
def test_workforce_pool_get_project_id_cloud_resource_manager_success(self):
    token_headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    token_request_data = {'grant_type': 'urn:ietf:params:oauth:grant-type:token-exchange', 'audience': self.WORKFORCE_AUDIENCE, 'requested_token_type': 'urn:ietf:params:oauth:token-type:access_token', 'subject_token': 'subject_token_0', 'subject_token_type': self.WORKFORCE_SUBJECT_TOKEN_TYPE, 'scope': 'scope1 scope2', 'options': urllib.parse.quote(json.dumps({'userProject': self.WORKFORCE_POOL_USER_PROJECT}))}
    request = self.make_mock_request(status=http_client.OK, data=self.SUCCESS_RESPONSE.copy(), cloud_resource_manager_status=http_client.OK, cloud_resource_manager_data=self.CLOUD_RESOURCE_MANAGER_SUCCESS_RESPONSE)
    credentials = self.make_workforce_pool_credentials(scopes=self.SCOPES, quota_project_id=self.QUOTA_PROJECT_ID, workforce_pool_user_project=self.WORKFORCE_POOL_USER_PROJECT)
    project_id = credentials.get_project_id(request)
    assert project_id == self.PROJECT_ID
    assert len(request.call_args_list) == 2
    self.assert_token_request_kwargs(request.call_args_list[0][1], token_headers, token_request_data)
    assert credentials.valid
    assert not credentials.expired
    assert credentials.token == self.SUCCESS_RESPONSE['access_token']
    self.assert_resource_manager_request_kwargs(request.call_args_list[1][1], self.WORKFORCE_POOL_USER_PROJECT, {'x-goog-user-project': self.QUOTA_PROJECT_ID, 'authorization': 'Bearer {}'.format(self.SUCCESS_RESPONSE['access_token'])})
    project_id = credentials.get_project_id(request)
    assert project_id == self.PROJECT_ID
    assert len(request.call_args_list) == 2