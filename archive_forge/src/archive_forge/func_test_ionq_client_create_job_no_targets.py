import contextlib
import datetime
import io
from unittest import mock
import requests
import pytest
import cirq_ionq as ionq
def test_ionq_client_create_job_no_targets():
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com', api_key='to_my_heart')
    with pytest.raises(AssertionError, match='neither were set'):
        _ = client.create_job(serialized_program=ionq.SerializedProgram(body={'job': 'mine'}, metadata={}, settings={}))