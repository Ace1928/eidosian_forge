import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
@mock.patch('requests.put')
def test_ionq_client_cancel_job_not_found(mock_put):
    mock_put.return_value.ok = False
    mock_put.return_value.status_code = requests.codes.not_found
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='to_my_heart', default_target='simulator')
    with pytest.raises(ionq.IonQNotFoundException, match='not find'):
        client.cancel_job('job_id')