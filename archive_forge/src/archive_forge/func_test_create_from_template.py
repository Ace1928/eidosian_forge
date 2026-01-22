import pytest
import numpy as np
import google.protobuf.text_format as text_format
import cirq
import cirq_google as cg
import cirq_google.api.v2 as v2
import cirq_google.engine.virtual_engine_factory as factory
def test_create_from_template():
    engine = factory.create_noiseless_virtual_engine_from_templates('sycamore', 'weber_2021_12_10_device_spec_for_grid_device.proto.txt')
    _test_processor(engine.get_processor('sycamore'))
    processor = factory.create_noiseless_virtual_processor_from_template('sycamore', 'weber_2021_12_10_device_spec_for_grid_device.proto.txt')
    _test_processor(processor)