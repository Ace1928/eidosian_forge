from unittest import TestCase
import macaroonbakery.httpbakery as httpbakery
import requests
from mock import patch
from httmock import HTTMock, response, urlmatch
def test_407_then_no_interaction_methods(self):
    client = httpbakery.Client(interaction_methods=[])
    with HTTMock(first_407_then_200), HTTMock(discharge_401):
        with self.assertRaises(httpbakery.InteractionError) as exc:
            requests.get(ID_PATH, cookies=client.cookies, auth=client.auth())
    self.assertEqual(str(exc.exception), 'cannot start interactive session: interaction required but not possible')