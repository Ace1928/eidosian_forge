import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
@mock.patch('requests.put')
def test_ionq_client_cancel_job_unauthorized(mock_put):
    mock_put.return_value.ok = False
    mock_put.return_value.status_code = requests.codes.unauthorized
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='to_my_heart', default_target='simulator')
    with pytest.raises(ionq.IonQException, match='Not authorized'):
        client.cancel_job('job_id')