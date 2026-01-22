from unittest import TestCase
import macaroonbakery.httpbakery as httpbakery
import requests
from mock import patch
from httmock import HTTMock, response, urlmatch
@urlmatch(path='.*/wait')
def wait_on_error(url, request):
    return {'status_code': 500, 'content': {'DischargeToken': discharge_token, 'Macaroon': discharged_macaroon}}