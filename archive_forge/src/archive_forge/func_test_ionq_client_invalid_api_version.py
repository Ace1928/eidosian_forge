import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
def test_ionq_client_invalid_api_version():
    with pytest.raises(AssertionError, match='is accepted'):
        _ = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='a', api_version='v0.0')
    with pytest.raises(AssertionError, match='0.0'):
        _ = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='a', api_version='v0.0')