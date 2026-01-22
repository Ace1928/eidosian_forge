import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
def test_ionq_client_attributes():
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='to_my_heart', default_target='qpu', max_retry_seconds=10, verbose=True)
    assert client.url == 'http://example.com/v0.3'
    assert client.headers == {'Authorization': 'apiKey to_my_heart', 'Content-Type': 'application/json', 'User-Agent': client._user_agent()}
    assert client.headers['User-Agent'].startswith('cirq/')
    assert client.default_target == 'qpu'
    assert client.max_retry_seconds == 10
    assert client.verbose is True