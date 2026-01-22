import datetime
from unittest import mock
import time
import numpy as np
import pytest
import duet
from google.protobuf import any_pb2, timestamp_pb2
from google.protobuf.text_format import Merge
import cirq
import cirq_google
import cirq_google as cg
from cirq_google.api import v1, v2
from cirq_google.engine import util
from cirq_google.cloud import quantum
from cirq_google.engine.engine import EngineContext
def test_run_sweep_with_multiple_processor_ids():
    engine = cg.Engine(project_id='proj', context=EngineContext(proto_version=cg.engine.engine.ProtoVersion.V2, enable_streaming=True))
    with pytest.raises(ValueError, match='multiple processors is no longer supported'):
        _ = engine.run_sweep(program=_CIRCUIT, params=[cirq.ParamResolver({'a': 1}), cirq.ParamResolver({'a': 2})], processor_ids=['mysim', 'mysim2'])