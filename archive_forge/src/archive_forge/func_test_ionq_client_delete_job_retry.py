import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
@mock.patch('requests.delete')
def test_ionq_client_delete_job_retry(mock_put):
    response1 = mock.MagicMock()
    response2 = mock.MagicMock()
    mock_put.side_effect = [response1, response2]
    response1.ok = False
    response1.status_code = requests.codes.service_unavailable
    response2.ok = True
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='to_my_heart', default_target='simulator')
    client.delete_job('job_id')
    assert mock_put.call_count == 2