import os
import requests
import testtools
from keystone.tests.common import auth as common_auth
def get_scoped_token_response(self, user):
    """Convenience method so that we can test authenticated requests.

        :param user: A dictionary with user information like 'username',
                     'password', 'domain_id'
        :returns: urllib3.Response object

        """
    body = self.build_authentication_request(username=user['name'], user_domain_name=user['domain_id'], password=user['password'], project_name=self.project_name, project_domain_id=self.project_domain_id)
    return requests.post(self.PUBLIC_URL + '/v3/auth/tokens', headers=self.request_headers, json=body)