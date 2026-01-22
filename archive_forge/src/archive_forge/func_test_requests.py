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
def test_requests():
    credentials, project_id = google.auth.default()
    credentials = google.auth.credentials.with_scopes_if_required(credentials, ['https://www.googleapis.com/auth/pubsub'])
    authed_session = google.auth.transport.requests.AuthorizedSession(credentials)
    with mock.patch.dict(os.environ, {environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE: 'true'}):
        authed_session.configure_mtls_channel()
    assert authed_session.is_mtls == mtls.has_default_client_cert_source()
    time.sleep(1)
    if authed_session.is_mtls:
        response = authed_session.get(MTLS_ENDPOINT.format(project_id))
    else:
        response = authed_session.get(REGULAR_ENDPOINT.format(project_id))
    assert response.ok