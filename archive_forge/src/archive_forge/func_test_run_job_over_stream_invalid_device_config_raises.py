import asyncio
import datetime
import os
from unittest import mock
import duet
import pytest
from google.api_core import exceptions
from google.protobuf import any_pb2
from google.protobuf.field_mask_pb2 import FieldMask
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.engine.engine_client import EngineClient, EngineException
import cirq_google.engine.stream_manager as engine_stream_manager
from cirq_google.cloud import quantum
@pytest.mark.parametrize('run_name, device_config_name', [('run1', ''), ('', 'device_config1')])
def test_run_job_over_stream_invalid_device_config_raises(run_name, device_config_name):
    client = EngineClient()
    with pytest.raises(ValueError, match='Cannot specify only one of `run_name` and `device_config_name`'):
        client.run_job_over_stream(project_id='proj', program_id='prog', code=any_pb2.Any(), job_id='job0', processor_id='mysim', run_context=any_pb2.Any(), run_name=run_name, device_config_name=device_config_name)