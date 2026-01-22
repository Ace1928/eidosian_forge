import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
@mock.patch('requests.post')
def test_ionq_client_create_job_retry(mock_post):
    response1 = mock.MagicMock()
    response2 = mock.MagicMock()
    mock_post.side_effect = [response1, response2]
    response1.ok = False
    response1.status_code = requests.codes.service_unavailable
    response2.ok = True
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='to_my_heart', default_target='simulator', verbose=True)
    test_stdout = io.StringIO()
    with contextlib.redirect_stdout(test_stdout):
        _ = client.create_job(serialized_program=ionq.SerializedProgram(body={'job': 'mine'}, metadata={}, settings={}))
    assert test_stdout.getvalue().strip() == 'Waiting 0.1 seconds before retrying.'
    assert mock_post.call_count == 2