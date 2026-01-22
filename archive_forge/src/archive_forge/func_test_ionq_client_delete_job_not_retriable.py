import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
@mock.patch('requests.delete')
def test_ionq_client_delete_job_not_retriable(mock_delete):
    mock_delete.return_value.ok = False
    mock_delete.return_value.status_code = requests.codes.not_implemented
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='to_my_heart', default_target='simulator')
    with pytest.raises(ionq.IonQException, match='Status: 501'):
        client.delete_job('job_id')