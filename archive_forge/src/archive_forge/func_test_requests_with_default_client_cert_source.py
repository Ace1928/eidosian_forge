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
def test_requests_with_default_client_cert_source():
    credentials, project_id = google.auth.default()
    credentials = google.auth.credentials.with_scopes_if_required(credentials, ['https://www.googleapis.com/auth/pubsub'])
    authed_session = google.auth.transport.requests.AuthorizedSession(credentials)
    if mtls.has_default_client_cert_source():
        with mock.patch.dict(os.environ, {environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE: 'true'}):
            authed_session.configure_mtls_channel(client_cert_callback=mtls.default_client_cert_source())
        assert authed_session.is_mtls
        time.sleep(1)
        response = authed_session.get(MTLS_ENDPOINT.format(project_id))
        assert response.ok