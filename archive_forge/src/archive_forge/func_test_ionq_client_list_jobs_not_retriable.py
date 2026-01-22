import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
@mock.patch('requests.get')
def test_ionq_client_list_jobs_not_retriable(mock_get):
    mock_get.return_value.ok = False
    mock_get.return_value.status_code = requests.codes.not_implemented
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='to_my_heart')
    with pytest.raises(ionq.IonQException, match='Status: 501'):
        _ = client.list_jobs()