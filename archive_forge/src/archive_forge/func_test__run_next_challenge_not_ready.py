import copy
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import reauth
def test__run_next_challenge_not_ready():
    challenges_response = copy.deepcopy(CHALLENGES_RESPONSE_TEMPLATE)
    challenges_response['challenges'][0]['status'] = 'STATUS_UNSPECIFIED'
    assert reauth._run_next_challenge(challenges_response, MOCK_REQUEST, 'token') is None