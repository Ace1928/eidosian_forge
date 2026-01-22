import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
def test_ionq_client_invalid_remote_host():
    for invalid_url in ('', 'url', 'http://', 'ftp://', 'http://'):
        with pytest.raises(AssertionError, match='not a valid url'):
            _ = ionq.ionq_client._IonQClient(remote_host=invalid_url, api_key='a')
        with pytest.raises(AssertionError, match=invalid_url):
            _ = ionq.ionq_client._IonQClient(remote_host=invalid_url, api_key='a')