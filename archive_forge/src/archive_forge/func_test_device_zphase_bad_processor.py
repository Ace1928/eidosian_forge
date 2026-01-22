import pytest
import numpy as np
import google.protobuf.text_format as text_format
import cirq
import cirq_google as cg
import cirq_google.api.v2 as v2
import cirq_google.engine.virtual_engine_factory as factory
def test_device_zphase_bad_processor():
    with pytest.raises(ValueError, match='no Z phase data is defined'):
        _ = factory.load_sample_device_zphase('bad_processor')