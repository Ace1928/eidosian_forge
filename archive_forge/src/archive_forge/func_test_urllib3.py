import json
import mock
import os
import time
from os import path
import google.auth
import google.auth.credentials
from google.auth import environment_vars
from google.auth.transport import mtls
import google.auth.transport.requests
import google.auth.transport.urllib3
def test_urllib3():
    credentials, project_id = google.auth.default()
    credentials = google.auth.credentials.with_scopes_if_required(credentials, ['https://www.googleapis.com/auth/pubsub'])
    authed_http = google.auth.transport.urllib3.AuthorizedHttp(credentials)
    with mock.patch.dict(os.environ, {environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE: 'true'}):
        is_mtls = authed_http.configure_mtls_channel()
    assert is_mtls == mtls.has_default_client_cert_source()
    time.sleep(1)
    if is_mtls:
        response = authed_http.request('GET', MTLS_ENDPOINT.format(project_id))
    else:
        response = authed_http.request('GET', REGULAR_ENDPOINT.format(project_id))
    assert response.status == 200