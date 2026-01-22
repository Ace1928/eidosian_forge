from datetime import datetime
import pytest
import google.auth
from google.auth import compute_engine
from google.auth import _helpers
from google.auth import exceptions
from google.auth import jwt
from google.auth.compute_engine import _metadata
import google.oauth2.id_token
def test_id_token_from_metadata(http_request):
    credentials = compute_engine.IDTokenCredentials(http_request, AUDIENCE, use_metadata_identity_endpoint=True)
    credentials.refresh(http_request)
    _, payload, _, _ = jwt._unverified_decode(credentials.token)
    assert credentials.valid
    assert payload['aud'] == AUDIENCE
    assert datetime.fromtimestamp(payload['exp']) == credentials.expiry