import pytest
from google.auth import jwt
import google.oauth2.id_token
def test_fetch_id_token(http_request):
    audience = 'https://pubsub.googleapis.com'
    token = google.oauth2.id_token.fetch_id_token(http_request, audience)
    _, payload, _, _ = jwt._unverified_decode(token)
    assert payload['aud'] == audience